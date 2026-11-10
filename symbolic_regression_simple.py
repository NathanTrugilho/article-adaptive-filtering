import numpy as np
import os
import sys
import time
from datetime import datetime
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


VARIABLE_NAMES = ["tau", "beta_", "N_", "sigmanu2", "sigmax2"]


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def section(title):
    print(f"\n{'=' * 60}", flush=True)
    print(f" {title}", flush=True)
    print(f"{'=' * 60}", flush=True)


def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def print_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- Metrics - {name} ---", flush=True)
    print(f"MSE:  {mse:.4e}")
    print(f"RMSE: {rmse:.4e}")
    print(f"MAE:  {mae:.4e}")
    print(f"R2:   {r2:.4f}", flush=True)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def select_best_by_validation(model, X_val, y_val):
    equations_df = model.equations_
    best_idx = None
    best_val_mse = np.inf
    val_mses = []

    for idx in equations_df.index:
        try:
            y_val_pred = model.predict(X_val, index=idx)
            if not np.all(np.isfinite(y_val_pred)):
                val_mses.append(np.nan)
                continue
            mse = mean_squared_error(y_val, y_val_pred)
            val_mses.append(mse)
            if mse < best_val_mse:
                best_val_mse = mse
                best_idx = idx
        except Exception:
            val_mses.append(np.nan)

    if best_idx is None:
        best_idx = model.get_best().name

    hof_with_val = equations_df.copy()
    hof_with_val["val_mse"] = val_mses
    best_row = hof_with_val.loc[best_idx]

    return best_idx, best_row, hof_with_val


def save_results(target_name, current_iters, metrics_train, metrics_val, metrics_test,
                 best_row, hof_df, model_path, train_time):
    filename = f"avaliacao_simple_{target_name}_{current_iters}iters.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f" SIMPLE MODEL EVALUATION - TARGET: {target_name}\n")
        f.write(f" CHECKPOINT: {current_iters} ITERATIONS\n")
        f.write("==================================================\n\n")

        f.write("[1] MODEL\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Directory: {model_path}\n")
        f.write("Split: train 70% | validation 10% | test 20%\n")
        f.write("Selection Criterion: Lowest Validation MSE\n")
        f.write(f"Cumulative training time: {format_duration(train_time)}\n")

        for title, m in [("TRAIN SET", metrics_train),
                         ("VALIDATION SET (SELECTION)", metrics_val),
                         ("TEST SET (EVALUATED WITH BEST VAL EQUATION)", metrics_test)]:
            f.write(f"\n[METRICS - {title}]\n")
            f.write("--------------------------------------------------\n")
            f.write(f"MSE:  {m['mse']:.4e}\n")
            f.write(f"RMSE: {m['rmse']:.4e}\n")
            f.write(f"MAE:  {m['mae']:.4e}\n")
            f.write(f"R2:   {m['r2']:.4f}\n")

        f.write("\n[SELECTED EQUATION (BY VALIDATION SET)]\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Complexity: {best_row['complexity']}\n")
        f.write(f"Val MSE:    {best_row['val_mse']:.4e}\n")
        f.write(f"Equation:   {best_row['equation']}\n")

        f.write("\n[HALL OF FAME WITH VALIDATION MSE]\n")
        f.write("--------------------------------------------------\n")
        cols = [c for c in ["complexity", "loss", "val_mse", "score", "equation"] if c in hof_df.columns]
        f.write(hof_df[cols].to_string(index=True) + "\n")

    log(f"Checkpoint results saved to: '{filename}'")

    history_file = f"historico_checkpoints_{target_name}.txt"
    with open(history_file, 'a', encoding='utf-8') as fh:
        fh.write(f"[{now()}] Iterações: {current_iters} | Tempo Acumulado: {format_duration(train_time)}\n")
        fh.write(f"  Selected Index: {best_row.name} (Complexity: {best_row['complexity']})\n")
        fh.write(f"  Train MSE: {metrics_train['mse']:.4e} | Val MSE: {metrics_val['mse']:.4e} | "
                 f"Test MSE: {metrics_test['mse']:.4e} | Test R2: {metrics_test['r2']:.4f}\n")
        fh.write(f"  Equação:   {best_row['equation']}\n")
        fh.write("-" * 80 + "\n")


def symbolic_regression_simple(X, y, target_name, total_checkpoints=5, step_iterations=1000):
    section(f"TARGET: {target_name}")

    log("Splitting data (70% train / 10% validation / 20% test)...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=28
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=28
    )

    log(f"Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")
    log(f"Variables: {', '.join(VARIABLE_NAMES)}")

    output_directory = f"simple_model_{target_name}"
    run_id = f"run_{target_name}"

    model = PySRRegressor(
        niterations=step_iterations,
        warm_start=True,
        output_directory=output_directory,
        run_id=run_id,
        populations=100,
        population_size=200,
        binary_operators=["+", "*", "-", "^"],
        unary_operators=["exp", "inv(x) = 1/x", "log10", "erf", "erfc"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        nested_constraints={
            "exp": {"exp": 0},
            "log10": {"log10": 0},
            "erf": {"erf": 0},
            "erfc": {"erfc": 0}
        },
        constraints={"^": (-1, 1)},
        parsimony=1e-3,
        alpha=3,
        ncycles_per_iteration=300,
        topn=24,
        model_selection="best",
        verbosity=True,
        progress=True,
        turbo=True,
        annealing=True,
        parallelism="multithreading"
    )

    total_iterations = total_checkpoints * step_iterations
    section(f"TRAINING STARTED - {target_name} ({total_checkpoints}x {step_iterations} = {total_iterations} iters)")

    cumulative_train_time = 0.0
    best_overall_val_mse = np.inf
    best_overall_info = None

    for ckpt in range(1, total_checkpoints + 1):
        current_iters = ckpt * step_iterations
        section(f"CHECKPOINT {ckpt}/{total_checkpoints} ({current_iters}/{total_iterations} ITERATIONS) - {target_name}")

        start_step = time.time()
        if ckpt == 1:
            model.fit(X_train, y_train, variable_names=VARIABLE_NAMES)
        else:
            model.fit(X_train, y_train)

        step_time = time.time() - start_step
        cumulative_train_time += step_time

        log(f"Checkpoint {ckpt} finished in: {format_duration(step_time)} "
            f"(Cumulative: {format_duration(cumulative_train_time)})")

        # 1. Seleciona a equação do Hall of Fame com menor MSE no conjunto de Validação
        best_val_idx, best_row, hof_with_val = select_best_by_validation(model, X_val, y_val)

        print(f"\n--- Best Equation Selected on Validation Set at {current_iters} iters ({target_name}) ---")
        print(f"Index: {best_val_idx} | Complexity: {best_row['complexity']} | Val MSE: {best_row['val_mse']:.4e}")
        print(f"Equation: {best_row['equation']}", flush=True)

        # 2. Usa a equação escolhida pela validação (index=best_val_idx) para predizer em Treino, Validação e Teste
        y_train_pred = model.predict(X_train, index=best_val_idx)
        y_val_pred = model.predict(X_val, index=best_val_idx)
        y_test_pred = model.predict(X_test, index=best_val_idx)

        metrics_train = print_metrics(y_train, y_train_pred, f"Train Set ({current_iters} iters)")
        metrics_val = print_metrics(y_val, y_val_pred, f"Validation Set ({current_iters} iters)")
        metrics_test = print_metrics(y_test, y_test_pred, f"Test Set ({current_iters} iters)")

        save_results(
            target_name,
            current_iters,
            metrics_train,
            metrics_val,
            metrics_test,
            best_row,
            hof_with_val,
            os.path.join(output_directory, run_id),
            cumulative_train_time
        )

        # Rastreia o melhor modelo global na validação entre todos os checkpoints
        if metrics_val["mse"] < best_overall_val_mse:
            best_overall_val_mse = metrics_val["mse"]
            best_overall_info = {
                "checkpoint_iters": current_iters,
                "equation": str(best_row["equation"]),
                "complexity": best_row["complexity"],
                "metrics_train": metrics_train,
                "metrics_val": metrics_val,
                "metrics_test": metrics_test
            }

    section(f"ALL CHECKPOINTS FINISHED - {target_name}")
    log(f"Total cumulative training time: {format_duration(cumulative_train_time)}")
    if best_overall_info is not None:
        log(f"Best overall Validation MSE occurred at {best_overall_info['checkpoint_iters']} iterations:")
        log(f"  Val MSE:  {best_overall_info['metrics_val']['mse']:.4e}")
        log(f"  Test MSE: {best_overall_info['metrics_test']['mse']:.4e} | Test R2: {best_overall_info['metrics_test']['r2']:.4f}")
        log(f"  Equation: {best_overall_info['equation']}")


if __name__ == '__main__':
    file = './resultsSMNLMS.csv'

    if not os.path.exists(file):
        print(f"Error: File '{file}' was not found.")
        print("Run the 'gerar_dataset.py' script first.")
        sys.exit(1)

    log(f"Loading data from '{file}'...")

    arr = np.loadtxt(file, delimiter=',', dtype=float, skiprows=1)
    log(f"Data loaded: {arr.shape[0]} samples, {arr.shape[1]} columns")

    X = arr[:, [0, 1, 2, 3, 4]]
    y_mse = arr[:, 5]

    total_start = time.time()

    NUM_CHECKPOINTS = 5
    STEP_ITERATIONS = 1000

    symbolic_regression_simple(
        X,
        y_mse,
        target_name="MSE",
        total_checkpoints=NUM_CHECKPOINTS,
        step_iterations=STEP_ITERATIONS
    )

    section("FINISHED")
    log(f"Total time: {format_duration(time.time() - total_start)}")