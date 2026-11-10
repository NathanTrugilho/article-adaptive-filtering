import numpy as np
import os
import sys
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


def save_results(target_name, model_params, metrics_train, metrics_val, metrics_test,
                 best_row, hof_df, model_path):
    filename = f"evaluation_{target_name}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f" LOADED MODEL EVALUATION - TARGET: {target_name}\n")
        f.write("==================================================\n\n")

        f.write("[1] LOADED MODEL\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Directory: {model_path}\n")
        f.write("Split: train 70% | validation 10% | test 20%\n")
        f.write("Selection Criterion: Lowest Validation MSE\n")

        f.write("\n[2] MODEL HYPERPARAMETERS\n")
        f.write("--------------------------------------------------\n")
        f.write(f"niterations (per step):  {model_params.get('niterations')}\n")
        f.write(f"warm_start:              {model_params.get('warm_start')}\n")
        f.write(f"populations:             {model_params.get('populations')}\n")
        f.write(f"population_size:         {model_params.get('population_size')}\n")
        f.write(f"parsimony:               {model_params.get('parsimony')}\n")
        f.write(f"alpha:                   {model_params.get('alpha')}\n")
        f.write(f"ncycles_per_iteration:   {model_params.get('ncycles_per_iteration')}\n")
        f.write(f"topn:                    {model_params.get('topn')}\n")

        for idx, (title, m) in enumerate([("TRAIN SET", metrics_train),
                                          ("VALIDATION SET (SELECTION)", metrics_val),
                                          ("TEST SET (EVALUATED WITH BEST VAL EQUATION)", metrics_test)], start=3):
            f.write(f"\n[{idx}] EVALUATION METRICS ({title})\n")
            f.write("--------------------------------------------------\n")
            f.write(f"MSE:  {m['mse']:.4e}\n")
            f.write(f"RMSE: {m['rmse']:.4e}\n")
            f.write(f"MAE:  {m['mae']:.4e}\n")
            f.write(f"R2:   {m['r2']:.4f}\n")

        f.write("\n[6] SELECTED EQUATION (BY VALIDATION SET)\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Hall of Fame Index: {best_row.name}\n")
        f.write(f"Complexity:         {best_row['complexity']}\n")
        f.write(f"Validation MSE:     {best_row['val_mse']:.4e}\n")
        f.write(f"Equation:           {best_row['equation']}\n")

        if hof_df is not None:
            f.write("\n[7] HALL OF FAME (PARETO FRONT WITH VALIDATION MSE)\n")
            f.write("--------------------------------------------------\n")
            cols = [c for c in ["complexity", "loss", "val_mse", "score", "equation"] if c in hof_df.columns]
            f.write(hof_df[cols].to_string(index=True) + "\n")

    log(f"Results successfully saved to: '{filename}'")


def evaluate_model(X, y, target_name, model_path=None):
    if model_path is None:
        model_path = os.path.join(f"simple_model_{target_name}", f"run_{target_name}")

    section(f"EVALUATING SAVED MODEL - TARGET: {target_name}")

    log("Splitting data (70% train / 10% validation / 20% test)...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=28
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=28
    )

    log(f"Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")

    if not os.path.exists(model_path):
        print(f"\nError: Model directory '{model_path}' was not found.")
        sys.exit(1)

    log(f"Loading model from '{model_path}'...")

    model = PySRRegressor.from_file(
        run_directory=model_path,
        extra_sympy_mappings={"inv": lambda x: 1 / x}
    )

    log("Model loaded successfully.")

    model_params = model.get_params()

    # 1. Seleciona a equação candidata com menor MSE no conjunto de Validação
    best_val_idx, best_row, hof_with_val = select_best_by_validation(model, X_val, y_val)

    print(f"\n--- Equation Selected by Validation Set ({target_name}) ---")
    print(f"Hall of Fame Index: {best_val_idx} | Complexity: {best_row['complexity']} | Val MSE: {best_row['val_mse']:.4e}")
    print(f"Equation: {best_row['equation']}", flush=True)

    # 2. Avalia os conjuntos de Treino, Validação e Teste usando a equação selecionada pela Validação
    section(f"METRICS - {target_name} (Selected Index: {best_val_idx})")
    metrics_train = print_metrics(y_train, model.predict(X_train, index=best_val_idx), "Train Set")
    metrics_val = print_metrics(y_val, model.predict(X_val, index=best_val_idx), "Validation Set")
    metrics_test = print_metrics(y_test, model.predict(X_test, index=best_val_idx), "Test Set")

    save_results(
        target_name,
        model_params,
        metrics_train,
        metrics_val,
        metrics_test,
        best_row,
        hof_with_val,
        model_path
    )


if __name__ == '__main__':
    file = './resultsSMNLMS.csv'
    target_name = "MSE"
    model_path = f"./simple_model_{target_name}/run_{target_name}"

    if not os.path.exists(file):
        print(f"Error: File '{file}' was not found.")
        print("Run the 'gerar_dataset.py' script first.")
        sys.exit(1)

    log(f"Loading data from '{file}'...")

    arr = np.loadtxt(file, delimiter=',', dtype=float, skiprows=1)
    log(f"Data loaded: {arr.shape[0]} samples, {arr.shape[1]} columns")

    X = arr[:, [0, 1, 2, 3, 4]]
    y_mse = arr[:, 5]

    evaluate_model(
        X,
        y_mse,
        target_name=target_name,
        model_path=model_path
    )

    section("EVALUATION FINISHED")