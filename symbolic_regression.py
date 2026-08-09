import numpy as np
import os
import sys
from pysr import PySRRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def print_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nMétricas - {name}:")
    print(f"MSE:  {mse:.4e}")
    print(f"RMSE: {rmse:.4e}")
    print(f"MAE:  {mae:.4e}")
    print(f"R2:   {r2:.4f}")

def RegressaoSimbolica(X, y, target_name):
    print(f"\n{'='*50}")
    print(f" Iniciando Busca para o Alvo: {target_name}")
    print(f"{'='*50}")

    # ======================
    # Separação de Dados
    # ======================
    # Separando 20% para teste final. Os 80% restantes farão K-Fold no Grid Search
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # ======================
    # Modelo Base (Grid Search)
    # ======================
    # Usando configurações rápidas para o grid search
    base_model = PySRRegressor(
        niterations=100,
        populations=100,
        population_size=200,
        binary_operators=["+", "*", "-", "^"],
        unary_operators=["exp", "inv(x) = 1/x", "log10", "erf", "erfc"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        nested_constraints={"exp": {"exp": 0}, "log10": {"log10": 0}, "erf": {"erf": 0}, "erfc": {"erfc": 0}},
        constraints={'^': (-1, 1)},
        model_selection='best',
        progress=False, # Desativado para não poluir o terminal no Grid Search
        verbosity=False,
        annealing=True,
        parallelism='multithreading'
    )

    # ======================
    # Hiperparâmetros (Grid)
    # ======================
    param_grid = {
        # O quão forte o modelo penaliza equações muito longas/complexas
        "parsimony": [0, 1e-4, 1e-3, 0.01],

        # Fator de temperatura (simulated annealing) ou probabilidade de aceitar equações piores
        "alpha": [3.17, 1, 5.0],

        "ncycles_per_iteration": [250, 380, 550],

        # Top N equações retornadas ou consideradas no pool de migração/seleção
        # 12 na primeira posição. 24 e 30 oferecem um pool maior de diversidade matemática.
        "topn": [6, 12, 24]
    }

    # ======================
    # K-Fold e Execução do Grid
    # ======================
    print(f"> Executando Grid Search com K-Fold para {target_name}...")
    cv = KFold(n_splits=5, shuffle=True, random_state=28)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        verbose=1,
        scoring="neg_mean_squared_error",
        n_jobs=1 # Deixe 1 para evitar conflito com o multithreading nativo do PySR
    )

    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    print("\n> Melhores hiperparâmetros encontrados (Grid):")
    print(best_params)

    # ======================
    # Modelo Final (Recriado)
    # ======================
    print("\n> Treinando o modelo final com os melhores parâmetros...")
    final_model = PySRRegressor(
        niterations=100,
		populations=100,
        population_size=100,
        binary_operators=["+", "*", "-", "^"],
        unary_operators=["exp", "inv(x) = 1/x", "log10", "erf", "erfc"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        nested_constraints={"exp": {"exp": 0}, "log10": {"log10": 0}, "erf": {"erf": 0}, "erfc": {"erfc": 0}},
        constraints={'^': (-1, 1)},
        model_selection='best',
        verbosity=True,
        progress=False,
        turbo=True,
        annealing=True,
        warm_start=True,
        parallelism='multithreading',
        output_directory=f"best_model_{target_name}",
        **best_params
    )

    final_model.fit(X_train, y_train)

    # ======================
    # Avaliação (Teste)
    # ======================
    y_test_pred = final_model.predict(X_test)
    print_metrics(y_test, y_test_pred, f"Test Set - {target_name}")

    # ======================
    # Resultado final
    # ======================
    print(f"\n> Melhor Equação Encontrada para {target_name}:")
    print(final_model.get_best().equation)


if __name__ == '__main__':
    file = './resultsSMNLMS.csv'
    
    # Verifica se o arquivo CSV já foi gerado
    if not os.path.exists(file):
        print(f"Erro: O arquivo '{file}' não foi encontrado.")
        print("Execute o script 'gerar_dataset.py' primeiro.")
        sys.exit(1)
        
    print('>> Carregando dados...')
    arr = np.loadtxt(file, delimiter=',', dtype=float)

    # Variáveis preditoras: tau, beta, N, sigmanu2, sigmax2
    X = arr[:, [0, 1, 2, 3, 4]]

    # Variáveis alvo
    y_mse = arr[:, 5]
    y_msd = arr[:, 6]
    y_pup = arr[:, 7]

    # Executa o pipeline completo para cada alvo
    RegressaoSimbolica(X, y_mse, target_name="MSE")
    #RegressaoSimbolica(X, y_msd, target_name="MSD")
    #RegressaoSimbolica(X, y_pup, target_name="Pup")

    print('\n>> FIM DE TODAS AS OTIMIZAÇÕES!')