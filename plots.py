import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def gerar_plot_modelo(X, y, target_name):

    # Repete EXATAMENTE o mesmo corte feito no treinamento (mesmo random_state=28)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # Carrega o modelo salvo em disco (precisa passar o extra_sympy_mappings para a predição funcionar se usar funções customizadas)
    model = PySRRegressor.from_file(
        run_directory= "best_model_MSE/20260810_125613_4Dyizm",
        extra_sympy_mappings={"inv": lambda x: 1 / x}
    )

    # Fazendo predições com o modelo carregado
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculando os R2
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # ==========================
    # Ordenação dos dados
    # ==========================
    # Pega os índices que ordenam o y_train do menor para o maior
    idx_train_sorted = np.argsort(y_train)
    y_train_sorted = y_train[idx_train_sorted]
    y_train_pred_sorted = y_train_pred[idx_train_sorted]
    x_train_indices = np.arange(len(y_train)) # Eixo X (0, 1, 2, ..., N)

    # Pega os índices que ordenam o y_test do menor para o maior
    idx_test_sorted = np.argsort(y_test)
    y_test_sorted = y_test[idx_test_sorted]
    y_test_pred_sorted = y_test_pred[idx_test_sorted]
    x_test_indices = np.arange(len(y_test)) # Eixo X (0, 1, 2, ..., N)

    # ==========================
    # Lógica de Plotagem
    # ==========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: TREINAMENTO
    axes[0].scatter(x_train_indices, y_train_sorted, color='blue', label='Real', alpha=0.7, edgecolors='none')
    axes[0].scatter(x_train_indices, y_train_pred_sorted, color='red', label='Predito', alpha=0.7, edgecolors='none')
    axes[0].set_title(f'Treinamento ({target_name})\n$R^2 = {r2_train:.4f}$', fontsize=14)
    axes[0].set_xlabel('Índice da Amostra (Ordenado pelo Valor Real)', fontsize=12)
    axes[0].set_ylabel('Valores', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.7)
    
    # Gráfico 2: TESTE
    axes[1].scatter(x_test_indices, y_test_sorted, color='blue', label='Real', alpha=0.7, edgecolors='none')
    axes[1].scatter(x_test_indices, y_test_pred_sorted, color='red', label='Predito', alpha=0.7, edgecolors='none')
    axes[1].set_title(f'Teste ({target_name})\n$R^2 = {r2_test:.4f}$', fontsize=14)
    axes[1].set_xlabel('Índice da Amostra (Ordenado pelo Valor Real)', fontsize=12)
    axes[1].set_ylabel('Valores', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    fig_name = f'grafico_real_vs_predito_{target_name}.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"> Gráfico salvo como '{fig_name}'")
    
    # Exibe o gráfico na tela
    plt.show()

if __name__ == '__main__':
    file = './resultsSMNLMS.csv'
    
    if not os.path.exists(file):
        print(f"Erro: O arquivo de dados '{file}' não foi encontrado.")
        sys.exit(1)
        
    print('>> Carregando dados para plotagem...')
    arr = np.loadtxt(file, delimiter=',', dtype=float)

    # Variáveis preditoras
    X = arr[:, [0, 1, 2, 3, 4]]

    # Alvos (Se você adicionar MSD e Pup depois, basta descomentar aqui)
    y_mse = arr[:, 5]
    # y_msd = arr[:, 6]
    # y_pup = arr[:, 7]

    # Gera os gráficos lendo os arquivos salvos pelo script de treinamento
    gerar_plot_modelo(X, y_mse, target_name="MSE")
    # gerar_plot_modelo(X, y_msd, target_name="MSD")
    # gerar_plot_modelo(X, y_pup, target_name="Pup")
    
    print("\n>> Fim da execução de plots.")