import numpy as np
import os
import sys
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def print_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nMetrics - {name}:")
    print(f"MSE:  {mse:.4e}")
    print(f"RMSE: {rmse:.4e}")
    print(f"MAE:  {mae:.4e}")
    print(f"R2:   {r2:.4f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def save_results(target_name, best_params, metrics, equation, model_path):
    filename = f"evaluation_{target_name}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f" MODEL EVALUATION - TARGET: {target_name}\n")
        f.write("==================================================\n\n")

        f.write("[1] LOADED MODEL\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Directory: {model_path}\n")

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

    print(f"\n> Results successfully saved to: '{filename}'")


def evaluate_model(X, y, target_name, model_path):
    print(f"\n{'=' * 50}")
    print(f" Evaluating saved model - Target: {target_name}")
    print(f"{'=' * 50}")

    # ======================
    # Data Split
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=28
    )

    print(f"> Training samples: {len(X_train)}")
    print(f"> Test samples:     {len(X_test)}")

    # ======================
    # Model Verification
    # ======================
    if not os.path.exists(model_path):
        print(f"\nError: Model directory '{model_path}' was not found.")
        sys.exit(1)

    # ======================
    # Load Model
    # ======================
    print(f"\n> Loading model from '{model_path}'...")

    model = PySRRegressor.from_file(run_directory=model_path)

    print("> Model loaded successfully.")

    # ======================
    # Grid Search Parameters
    # ======================
    model_params = model.get_params()

    best_params = {
        "parsimony": model_params["parsimony"],
        "alpha": model_params["alpha"],
        "ncycles_per_iteration": model_params["ncycles_per_iteration"],
        "topn": model_params["topn"]
    }

    print("\n> Grid Search parameters used:")

    for param, value in best_params.items():
        print(f"{param}: {value}")

    # ======================
    # Best Equation
    # ======================
    best_equation = model.get_best().equation

    print("\n> Loaded model equation:")
    print(best_equation)

    # ======================
    # Prediction
    # ======================
    print("\n> Running predictions on Test Set...")

    y_test_pred = model.predict(X_test)

    # ======================
    # Evaluation
    # ======================
    metrics = print_metrics(
        y_test,
        y_test_pred,
        f"Test Set - {target_name}"
    )

    # ======================
    # Save Results
    # ======================
    save_results(
        target_name,
        best_params,
        metrics,
        str(best_equation),
        model_path
    )


if __name__ == '__main__':

    # ======================
    # Files
    # ======================
    file = './resultsSMNLMS.csv'

    model_path = './best_model_MSE/20260826_110757_a3pBEd/'

    target_name = "MSE"

    # ======================
    # Dataset Verification
    # ======================
    if not os.path.exists(file):
        print(f"Error: File '{file}' was not found.")
        print("Run the 'gerar_dataset.py' script first.")
        sys.exit(1)

    # ======================
    # Load Dataset
    # ======================
    print('>> Loading data...')

    arr = np.loadtxt(
        file,
        delimiter=',',
        dtype=float
    )

    # Predictor variables:
    # tau, beta, N, sigmanu2, sigmax2
    X = arr[:, [0, 1, 2, 3, 4]]

    # Target variable
    y_mse = arr[:, 5]

    # ======================
    # Model Evaluation
    # ======================
    evaluate_model(
        X,
        y_mse,
        target_name=target_name,
        model_path=model_path
    )

    print('\n>> EVALUATION FINISHED!')