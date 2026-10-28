import numpy as np
import os
import sys
import time
from datetime import datetime
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Variable names used by PySR in the equations (instead of x0, x1, ...).
# NOTE: "beta" and "N" are reserved SymPy function names, so PySR rejects them.
#       They are renamed to "beta_" and "N_".
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


def save_results(target_name, metrics_train, metrics_val, metrics_test, equation, model_path, train_time):
    filename = f"avaliacao_simple_{target_name}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f" SIMPLE MODEL EVALUATION - TARGET: {target_name}\n")
        f.write("==================================================\n\n")

        f.write("[1] MODEL\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Directory: {model_path}\n")
        f.write("Split: train 70% | validation 10% | test 20%\n")
        f.write(f"Training time: {format_duration(train_time)}\n")

        for title, m in [("TRAIN SET", metrics_train),
                         ("VALIDATION SET", metrics_val),
                         ("TEST SET", metrics_test)]:
            f.write(f"\n[METRICS - {title}]\n")
            f.write("--------------------------------------------------\n")
            f.write(f"MSE:  {m['mse']:.4e}\n")
            f.write(f"RMSE: {m['rmse']:.4e}\n")
            f.write(f"MAE:  {m['mae']:.4e}\n")
            f.write(f"R2:   {m['r2']:.4f}\n")

        f.write("\n[MODEL EQUATION]\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{equation}\n")

    log(f"Results saved to: '{filename}'")


def symbolic_regression_simple(X, y, target_name):
    section(f"TARGET: {target_name}")

    # ======================
    # Data Split: 70% train / 10% validation / 20% test
    # ======================
    log("Splitting data (70% train / 10% validation / 20% test)...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=28
    )

    # 10% of total = 0.125 of the remaining 80%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.125,
        random_state=28
    )

    log(f"Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")
    log(f"Variables: {', '.join(VARIABLE_NAMES)}")

    # ======================
    # Model (fixed hyperparameters)
    # ======================
    output_directory = f"simple_model_{target_name}"

    model = PySRRegressor(
        niterations=100,
        populations=30,
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
        parallelism="multithreading",
        output_directory=output_directory
    )

    # ======================
    # Training
    # ======================
    section(f"TRAINING STARTED - {target_name}")
    log(f"niterations={model.niterations} | populations={model.populations} | "
        f"population_size={model.population_size}")
    log("Note: the first run may take a while to compile Julia before the search begins.")

    start = time.time()
    model.fit(X_train, y_train, variable_names=VARIABLE_NAMES)
    train_time = time.time() - start

    section(f"TRAINING FINISHED - {target_name}")
    log(f"Training time: {format_duration(train_time)}")
    log(f"Model saved in: '{output_directory}'")

    # ======================
    # Evaluation
    # ======================
    section(f"EVALUATION - {target_name}")
    metrics_train = print_metrics(y_train, model.predict(X_train), "Train Set")
    metrics_val = print_metrics(y_val, model.predict(X_val), "Validation Set")
    metrics_test = print_metrics(y_test, model.predict(X_test), "Test Set")

    # ======================
    # Best Equation
    # ======================
    best_equation = model.get_best().equation

    print(f"\n--- Best Equation ({target_name}) ---")
    print(best_equation, flush=True)

    # ======================
    # Save Results
    # ======================
    save_results(
        target_name,
        metrics_train,
        metrics_val,
        metrics_test,
        str(best_equation),
        output_directory,
        train_time
    )


if __name__ == '__main__':

    # ======================
    # Dataset
    # ======================
    file = './resultsSMNLMS.csv'

    if not os.path.exists(file):
        print(f"Error: File '{file}' was not found.")
        print("Run the 'gerar_dataset.py' script first.")
        sys.exit(1)

    log(f"Loading data from '{file}'...")

    arr = np.loadtxt(
        file,
        delimiter=',',
        dtype=float,
        skiprows=1  # skip header: tau,beta,N,sigmanu2,sigmax2,MSE,MSD,Pup
    )

    log(f"Data loaded: {arr.shape[0]} samples, {arr.shape[1]} columns")

    # Predictor variables:
    # tau, beta, N, sigmanu2, sigmax2
    X = arr[:, [0, 1, 2, 3, 4]]

    # Target variables
    y_mse = arr[:, 5]
    # y_msd = arr[:, 6]
    # y_pup = arr[:, 7]

    # ======================
    # Regression
    # ======================
    total_start = time.time()

    symbolic_regression_simple(X, y_mse, target_name="MSE")

    # symbolic_regression_simple(X, y_msd, target_name="MSD")

    # symbolic_regression_simple(X, y_pup, target_name="Pup")

    section("FINISHED")
    log(f"Total time: {format_duration(time.time() - total_start)}")
