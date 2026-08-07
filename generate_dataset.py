import numpy as np
import pandas as pd

def evaluateEmpiricalQuantitiesSMNLMS(tauVector , betaVector , NVector , sigmanu2Vector , sigmax2Vector , numberOfIterations , numberOfLastIterations , numberOfRepeats):
	
	MSE = np.zeros((len(tauVector),len(betaVector),len(NVector),len(sigmanu2Vector),len(sigmax2Vector)))
	MSD = np.zeros((len(tauVector),len(betaVector),len(NVector),len(sigmanu2Vector),len(sigmax2Vector)))
	Pup = np.zeros((len(tauVector),len(betaVector),len(NVector),len(sigmanu2Vector),len(sigmax2Vector)))

	for tauIndex in range(len(tauVector)):
		for betaIndex in range(len(betaVector)):
			for NIndex in range(len(NVector)):
				for sigmanu2Index in range(len(sigmanu2Vector)):
					for sigmax2Index in range(len(sigmax2Vector)):
						print(str(tauIndex + 1) + '/' + str(len(tauVector)) + ' - ' + str(betaIndex + 1) + '/' + str(len(betaVector)) + ' - ' + str(NIndex + 1) + '/' + str(len(NVector)) + ' - ' + str(sigmanu2Index + 1) + '/' + str(len(sigmanu2Vector)) + ' - ' + str(sigmax2Index + 1) +  '/' + str(len(sigmax2Vector)))
						tau = tauVector[tauIndex]
						beta = betaVector[betaIndex]
						N = NVector[NIndex]
						sigmanu2 = sigmanu2Vector[sigmanu2Index]
						sigmax2 = sigmax2Vector[sigmax2Index]

						for repeat in range(numberOfRepeats):
							wk = np.zeros((N,1))
							w0 = np.random.randn(N,1)
							x  = np.sqrt(sigmax2) * np.random.randn( numberOfIterations + N - 1, 1 )
							d  = np.convolve(w0[:,0], x[:,0])
							d  += np.sqrt( sigmanu2 ) * np.random.randn(len(d))
							gamma = np.sqrt(tau * sigmanu2)

							for k in range(N, numberOfIterations + N - 1):
								iteration = k - N + 1
								xk = x[k:k-N:-1]
								yk = np.dot(wk.T, xk)
								ek = d[k] - yk
								
								if abs(ek) > gamma:
									mu = 1 - gamma/abs(ek)
									wk = wk + beta * mu / (np.dot(xk.T, xk)) * ek * xk

									if iteration > numberOfIterations - numberOfLastIterations + 1:
										Pup[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index] += 1 / (numberOfRepeats * numberOfLastIterations)

								if iteration > numberOfIterations - numberOfLastIterations + 1:
									MSD[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index] += np.linalg.norm(wk - w0) ** 2 / (numberOfRepeats * numberOfLastIterations)
									MSE[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index] += ek.item() ** 2 / (numberOfRepeats * numberOfLastIterations)
	return MSE,MSD,Pup

if __name__ == '__main__':
    # Obs: Foram mantidas as últimas atribuições feitas no código original para os parâmetros
    tauVector = np.arange(0, 2)
    betaVector = np.arange(0.1, 1.1, 0.1)
    NVector = [10]
    sigmanu2Vector = np.power(10,np.arange(-6, -1), dtype=float)
    sigmax2Vector = np.power(10,np.arange(1, 2), dtype=float)
    numberOfIterations = 50
    numberOfLastIterations = 1000
    numberOfRepeats = 100

    print('>> Gerando dados SMNLMS...')
    MSE,MSD,Pup = evaluateEmpiricalQuantitiesSMNLMS(tauVector,betaVector,NVector,sigmanu2Vector,sigmax2Vector,numberOfIterations,numberOfLastIterations,numberOfRepeats)

    data = {
        'tau': [],
        'beta': [],
        'N': [],
        'sigmanu2': [],
        'sigmax2': [],
        'MSE': [],
        'MSD': [],
        'Pup': [],
    }

    for tauIndex in range(len(tauVector)):
            for betaIndex in range(len(betaVector)):
                for NIndex in range(len(NVector)):
                    for sigmanu2Index in range(len(sigmanu2Vector)):
                        for sigmax2Index in range(len(sigmax2Vector)):
                            data['tau'].append(tauVector[tauIndex])
                            data['beta'].append(betaVector[betaIndex])
                            data['N'].append(NVector[NIndex])
                            data['sigmanu2'].append(sigmanu2Vector[sigmanu2Index])
                            data['sigmax2'].append(sigmax2Vector[sigmax2Index])
                            data['MSE'].append(MSE[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index])
                            data['MSD'].append(MSD[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index])
                            data['Pup'].append(Pup[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index])
                            
    file = './resultsSMNLMS.csv'

    print('>> Gerando arquivo csv gerado SMNLMS')
    df = pd.DataFrame(data)
    df.to_csv(file, index=False, header=False)
    print('>> FIM DA GERAÇÃO DE DADOS!')