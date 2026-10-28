import numpy as np
import os
import sys
import time
from datetime import datetime
from pysr import PySRRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
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


def save_results(target_name, best_params, metrics, equation, model_path, grid_time, train_time):
    filename = f"avaliacao_{target_name}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f" MODEL EVALUATION - TARGET: {target_name}\n")
        f.write("==================================================\n\n")

        f.write("[1] MODEL\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Directory: {model_path}\n")
        f.write(f"Variables: {', '.join(VARIABLE_NAMES)}\n")
        f.write(f"Grid Search time:   {format_duration(grid_time)}\n")
        f.write(f"Final training time: {format_duration(train_time)}\n")

        f.write("\n[2] BEST HYPERPARAMETERS (GRID SEARCH)\n")
        f.write("--------------------------------------------------\n")
        f.write(f"parsimony:               {best_params['parsimony']}\n")
        f.write(f"alpha:                   {best_params['alpha']}\n")
        f.write(f"ncycles_per_iteration:   {best_params['ncycles_per_iteration']}\n")
        f.write(f"topn:                    {best_params['topn']}\n")

        f.write("\n[3] EVALUATION METRICS (TEST SET)\n")
        f.write("--------------------------------------------------\n")
        f.write(f"MSE:  {metrics['mse']:.4e}\n")
        f.write(f"RMSE: {metrics['rmse']:.4e}\n")
        f.write(f"MAE:  {metrics['mae']:.4e}\n")
        f.write(f"R2:   {metrics['r2']:.4f}\n")

        f.write("\n[4] MODEL EQUATION\n")
        f.write("--------------------------------------------------\n")
        f.write(f"{equation}\n")

    log(f"Results saved to: '{filename}'")


def symbolic_regression(X, y, target_name):
    section(f"TARGET: {target_name}")

    # ======================
    # Data Split
    # ======================
    log("Splitting data (80% train / 20% test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=28
    )

    log(f"Train: {len(y_train)} | Test: {len(y_test)}")
    log(f"Variables: {', '.join(VARIABLE_NAMES)}")

    # ======================
    # Base Model
    # ======================
    base_model = PySRRegressor(
        niterations=20,
        populations=10,
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
        model_selection="best",
        progress=False,
        verbosity=False,
        annealing=True,
        parallelism="multithreading"
    )

    # ======================
    # Hyperparameter Grid
    # ======================
    param_grid = {
        "parsimony": [1e-4, 1e-3, 1e-2],
        "alpha": [2, 3, 4],
        "ncycles_per_iteration": [150, 300, 450],
        "topn": [12, 24, 36]
    }

    # ======================
    # Grid Search
    # ======================
    n_splits = 5
    n_combinations = int(np.prod([len(v) for v in param_grid.values()]))

    section(f"GRID SEARCH STARTED - {target_name}")
    log(f"{n_combinations} combinations x {n_splits} folds = "
        f"{n_combinations * n_splits} fits")
    log("Note: the first run may take a while to compile Julia before the search begins.")

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=28
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        verbose=3,
        scoring="neg_mean_squared_error",
        n_jobs=1
    )

    grid_start = time.time()
    grid.fit(X_train, y_train, variable_names=VARIABLE_NAMES)
    grid_time = time.time() - grid_start

    best_params = grid.best_params_

    section(f"GRID SEARCH FINISHED - {target_name}")
    log(f"Grid Search time: {format_duration(grid_time)}")
    log(f"Best CV score (neg MSE): {grid.best_score_:.4e}")
    log("Best hyperparameters found:")
    for param, value in best_params.items():
        print(f"    {param}: {value}", flush=True)

    # ======================
    # Final Model
    # ======================
    output_directory = f"best_model_{target_name}"

    final_model = PySRRegressor(
        niterations=10000,
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
        model_selection="best",
        verbosity=True,
        progress=True,
        turbo=True,
        annealing=True,
        warm_start=True,
        parallelism="multithreading",
        output_directory=output_directory,
        **best_params
    )

    section(f"FINAL TRAINING STARTED - {target_name}")
    log(f"niterations={final_model.niterations} | populations={final_model.populations} | "
        f"population_size={final_model.population_size}")

    train_start = time.time()
    final_model.fit(X_train, y_train, variable_names=VARIABLE_NAMES)
    train_time = time.time() - train_start

    # ======================
    # Model Path
    # ======================
    model_path = output_directory

    section(f"FINAL TRAINING FINISHED - {target_name}")
    log(f"Training time: {format_duration(train_time)}")
    log(f"Model saved in: '{model_path}'")

    # ======================
    # Test Evaluation
    # ======================
    section(f"EVALUATION - {target_name}")
    y_test_pred = final_model.predict(X_test)

    metrics = print_metrics(
        y_test,
        y_test_pred,
        "Test Set"
    )

    # ======================
    # Best Equation
    # ======================
    best_equation = final_model.get_best().equation

    print(f"\n--- Best Equation ({target_name}) ---")
    print(best_equation, flush=True)

    # ======================
    # Save Results
    # ======================
    save_results(
        target_name,
        best_params,
        metrics,
        str(best_equation),
        model_path,
        grid_time,
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

    symbolic_regression(
        X,
        y_mse,
        target_name="MSE"
    )

    # symbolic_regression(
    #     X,
    #     y_msd,
    #     target_name="MSD"
    # )

    # symbolic_regression(
    #     X,
    #     y_pup,
    #     target_name="Pup"
    # )

    section("ALL OPTIMIZATIONS FINISHED!")
    log(f"Total time: {format_duration(time.time() - total_start)}")