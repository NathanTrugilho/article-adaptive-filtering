import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def plot_model(X, y, target_name, model_path):
    print(f"\n{'=' * 50}")
    print(f" Generating plot for target: {target_name}")
    print(f"{'=' * 50}")

    # ======================
    # Model Verification
    # ======================
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' was not found.")
        sys.exit(1)

    # ======================
    # Data Split
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=28
    )

    # ======================
    # Load Model
    # ======================
    print(f"> Loading model from '{model_path}'...")

    model = PySRRegressor.from_file(
        run_directory=model_path,
        extra_sympy_mappings={
            "inv": lambda x: 1 / x
        }
    )

    print("> Model loaded successfully.")

    # ======================
    # Predictions
    # ======================
    print("> Running predictions...")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ======================
    # R2
    # ======================
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"\nR2 Training: {r2_train:.4f}")
    print(f"R2 Test:     {r2_test:.4f}")

    # ======================
    # Sort Data
    # ======================
    idx_train_sorted = np.argsort(y_train)

    y_train_sorted = y_train[idx_train_sorted]
    y_train_pred_sorted = y_train_pred[idx_train_sorted]

    x_train_indices = np.arange(len(y_train))

    idx_test_sorted = np.argsort(y_test)

    y_test_sorted = y_test[idx_test_sorted]
    y_test_pred_sorted = y_test_pred[idx_test_sorted]

    x_test_indices = np.arange(len(y_test))

    # ======================
    # Plot
    # ======================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # Training
    axes[0].scatter(
        x_train_indices,
        y_train_sorted,
        color='blue',
        label='Actual',
        alpha=0.7,
        edgecolors='none'
    )

    axes[0].scatter(
        x_train_indices,
        y_train_pred_sorted,
        color='red',
        label='Predicted',
        alpha=0.7,
        edgecolors='none'
    )

    axes[0].set_title(
        f'Training ({target_name})\n$R^2 = {r2_train:.4f}$',
        fontsize=14
    )

    axes[0].set_xlabel(
        'Sample Index (Sorted by Actual Value)',
        fontsize=12
    )

    axes[0].set_ylabel(
        'Values',
        fontsize=12
    )

    axes[0].legend()
    axes[0].grid(
        True,
        linestyle=':',
        alpha=0.7
    )

    # Test
    axes[1].scatter(
        x_test_indices,
        y_test_sorted,
        color='blue',
        label='Actual',
        alpha=0.7,
        edgecolors='none'
    )

    axes[1].scatter(
        x_test_indices,
        y_test_pred_sorted,
        color='red',
        label='Predicted',
        alpha=0.7,
        edgecolors='none'
    )

    axes[1].set_title(
        f'Test ({target_name})\n$R^2 = {r2_test:.4f}$',
        fontsize=14
    )

    axes[1].set_xlabel(
        'Sample Index (Sorted by Actual Value)',
        fontsize=12
    )

    axes[1].set_ylabel(
        'Values',
        fontsize=12
    )

    axes[1].legend()
    axes[1].grid(
        True,
        linestyle=':',
        alpha=0.7
    )

    plt.tight_layout()

    # ======================
    # Save Figure
    # ======================
    fig_name = f'actual_vs_predicted_{target_name}.png'

    plt.savefig(
        fig_name,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"\n> Plot saved as '{fig_name}'")

    plt.show()


if __name__ == '__main__':

    # ======================
    # Files
    # ======================
    file = './resultsSMNLMS.csv'

    model_path = './best_model_MSE/20260826_110757_a3pBEd/'

    target_name = 'MSE'

    # ======================
    # Dataset Verification
    # ======================
    if not os.path.exists(file):
        print(f"Error: Dataset file '{file}' was not found.")
        sys.exit(1)

    # ======================
    # Load Dataset
    # ======================
    print('>> Loading data for plotting...')

    arr = np.loadtxt(
        file,
        delimiter=',',
        dtype=float
    )

    # Predictor variables:
    # tau, beta, N, sigmanu2, sigmax2
    X = arr[:, [0, 1, 2, 3, 4]]

    # Target variables
    y_mse = arr[:, 5]
    # y_msd = arr[:, 6]
    # y_pup = arr[:, 7]

    # ======================
    # Generate Plot
    # ======================
    plot_model(
        X,
        y_mse,
        target_name=target_name,
        model_path=model_path
    )

    # plot_model(
    #     X,
    #     y_msd,
    #     target_name='MSD',
    #     model_path='./best_model_MSD/model_directory/'
    # )

    # plot_model(
    #     X,
    #     y_pup,
    #     target_name='Pup',
    #     model_path='./best_model_Pup/model_directory/'
    # )

    print("\n>> Plot generation finished.")