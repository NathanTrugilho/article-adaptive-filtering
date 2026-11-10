import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def select_best_by_validation(model, X_val, y_val):
    """
    Avalia todas as equações da fronteira de Pareto (model.equations_) no conjunto
    de validação e retorna o índice daquela com menor MSE de validação.
    """
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

    return best_idx, best_row


def plot_model(X, y, target_name, model_path=None):
    if model_path is None:
        model_path = os.path.join(f"simple_model_{target_name}", f"run_{target_name}")

    print(f"\n{'=' * 60}")
    print(f" Generating plot for target: {target_name}")
    print(f"{'=' * 60}")

    # ======================
    # Model Verification
    # ======================
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' was not found.")
        sys.exit(1)

    # ======================
    # Data Split: 70% train / 10% validation / 20% test (idêntico ao treino)
    # ======================
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=28
    )

    # 10% do total = 0.125 dos 80% restantes
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.125,
        random_state=28
    )

    print(f"> Samples -> Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")

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
    # Select Best Equation by Validation Set
    # ======================
    best_val_idx, best_row = select_best_by_validation(model, X_val, y_val)

    print(f"\n> Equation selected by Validation Set:")
    print(f"  Index: {best_val_idx} | Complexity: {best_row['complexity']} | Val MSE: {best_row['val_mse']:.4e}")
    print(f"  Equation: {best_row['equation']}")

    # ======================
    # Predictions (usando a equação escolhida pela validação)
    # ======================
    print("\n> Running predictions...")

    y_train_pred = model.predict(X_train, index=best_val_idx)
    y_val_pred = model.predict(X_val, index=best_val_idx)
    y_test_pred = model.predict(X_test, index=best_val_idx)

    # ======================
    # R2 Scores
    # ======================
    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"\nR2 Training:   {r2_train:.4f}")
    print(f"R2 Validation: {r2_val:.4f}")
    print(f"R2 Test:       {r2_test:.4f}")

    # ======================
    # Plot (Training, Validation & Test)
    # ======================
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5)
    )

    datasets = [
        ("Training", y_train, y_train_pred, r2_train),
        ("Validation (Selection)", y_val, y_val_pred, r2_val),
        ("Test", y_test, y_test_pred, r2_test)
    ]

    for ax, (split_title, y_true_split, y_pred_split, r2_split) in zip(axes, datasets):
        idx_sorted = np.argsort(y_true_split)
        y_true_sorted = y_true_split[idx_sorted]
        y_pred_sorted = y_pred_split[idx_sorted]
        x_indices = np.arange(len(y_true_split))

        ax.scatter(
            x_indices,
            y_true_sorted,
            color='blue',
            label='Actual',
            alpha=0.7,
            edgecolors='none'
        )

        ax.scatter(
            x_indices,
            y_pred_sorted,
            color='red',
            label='Predicted',
            alpha=0.7,
            edgecolors='none'
        )

        ax.set_title(
            f'{split_title} ({target_name})\n$R^2 = {r2_split:.4f}$',
            fontsize=13
        )

        ax.set_xlabel(
            'Sample Index (Sorted by Actual Value)',
            fontsize=11
        )

        ax.set_ylabel(
            'Values',
            fontsize=11
        )

        ax.legend()
        ax.grid(
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
    target_name = 'MSE'

    # Caminho padrão gerado pelo script de checkpoints com run_id fixo
    model_path = f'./simple_model_{target_name}/run_{target_name}'

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
        dtype=float,
        skiprows=1  # Pula o cabeçalho: tau,beta,N,sigmanu2,sigmax2,MSE,MSD,Pup
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
    #     model_path='./simple_model_MSD/run_MSD'
    # )

    # plot_model(
    #     X,
    #     y_pup,
    #     target_name='Pup',
    #     model_path='./simple_model_Pup/run_Pup'
    # )

    print("\n>> Plot generation finished.")