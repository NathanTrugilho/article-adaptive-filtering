import numpy as np
from pysr import *
import sys
import os

def RegressaoSimbolica(X,y):
	model = PySRRegressor(
		parsimony = 0.0001, # padrão = 0.0032
		progress=False,
		niterations=100,
		weight_randomize=0.001, # default: 0.00023
		populations=300,
		population_size=100,
		model_selection='score', # 'accuracy', 'best', or 'score'
		nested_constraints={"exp":{"exp": 0},"log10":{"log10": 0},"erf":{"erf": 0},"erfc":{"erfc": 0}},
		binary_operators=["+", "*","-","^"],
		unary_operators=[
			"exp",
			"inv(x) = 1/x",
			"log10",
			"erf",
			"erfc",
		],
		extra_sympy_mappings={"inv": lambda x: 1 / x})

	model.fit(X, y)
	print(model)

	best_idx = model.equations_.query(
		f"loss < {2 * model.equations_.loss.min()}"
	).score.idxmax()
	model.sympy(best_idx)

	model.get_best().equation

if __name__ == '__main__':
    file = './resultsSMNLMS.csv'
    
    # Verifica se o arquivo CSV já foi gerado
    if not os.path.exists(file):
        print(f"Erro: O arquivo '{file}' não foi encontrado.")
        print("Execute o script 'gerar_dataset.py' primeiro.")
        sys.exit(1)
        
    print('>> Iniciando Regressão Simbólica...')

    arr = np.loadtxt( file , delimiter = ',' , dtype = float)

    #tau beta N sigmanu2 sigmax2
    X = arr[ : ,  [ 0 , 1 , 2 , 3 , 4 ] ]

    print('>> MSE')
    RegressaoSimbolica(X, arr[ : , 5 ])
    print('>> MSD')
    RegressaoSimbolica(X, arr[ : , 6 ])
    print('>> Pup')
    RegressaoSimbolica(X, arr[ : , 7 ])

    print('>> FIM!')