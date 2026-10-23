import numpy as np
import pandas as pd

def evaluate_empirical_quantities_smnlms(tau_vector, beta_vector, n_vector, sigmanu2_vector, sigmax2_vector, number_of_iterations, number_of_last_iterations, number_of_repeats):
    mse = np.zeros((len(tau_vector), len(beta_vector), len(n_vector), len(sigmanu2_vector), len(sigmax2_vector)))
    msd = np.zeros((len(tau_vector), len(beta_vector), len(n_vector), len(sigmanu2_vector), len(sigmax2_vector)))
    pup = np.zeros((len(tau_vector), len(beta_vector), len(n_vector), len(sigmanu2_vector), len(sigmax2_vector)))

    total_combinations = len(tau_vector) * len(beta_vector) * len(n_vector) * len(sigmanu2_vector) * len(sigmax2_vector)
    combination_count = 0

    for tau_index in range(len(tau_vector)):
        for beta_index in range(len(beta_vector)):
            for n_index in range(len(n_vector)):
                for sigmanu2_index in range(len(sigmanu2_vector)):
                    for sigmax2_index in range(len(sigmax2_vector)):
                        combination_count += 1
                        print(f"Combination {combination_count}/{total_combinations}")

                        tau = tau_vector[tau_index]
                        beta = beta_vector[beta_index]
                        n_filter = n_vector[n_index]
                        sigmanu2 = sigmanu2_vector[sigmanu2_index]
                        sigmax2 = sigmax2_vector[sigmax2_index]

                        for repeat in range(number_of_repeats):
                            wk = np.zeros((n_filter, 1))
                            w0 = np.random.randn(n_filter, 1)
                            x = np.sqrt(sigmax2) * np.random.randn(number_of_iterations + n_filter - 1, 1)
                            d = np.convolve(w0[:, 0], x[:, 0])
                            d += np.sqrt(sigmanu2) * np.random.randn(len(d))
                            gamma = np.sqrt(tau * sigmanu2)

                            for k in range(n_filter, number_of_iterations + n_filter - 1):
                                iteration = k - n_filter + 1
                                xk = x[k:k - n_filter:-1]
                                yk = np.dot(wk.T, xk)
                                ek = d[k] - yk

                                if abs(ek) > gamma:
                                    mu = 1.0 - gamma / abs(ek)
                                    wk = wk + (beta * mu / np.dot(xk.T, xk)) * ek * xk

                                    if iteration > number_of_iterations - number_of_last_iterations:
                                        pup[tau_index, beta_index, n_index, sigmanu2_index, sigmax2_index] += 1.0 / (number_of_repeats * number_of_last_iterations)

                                if iteration > number_of_iterations - number_of_last_iterations:
                                    norm_diff_squared = np.linalg.norm(wk - w0) ** 2
                                    msd[tau_index, beta_index, n_index, sigmanu2_index, sigmax2_index] += norm_diff_squared / (number_of_repeats * number_of_last_iterations)
                                    mse[tau_index, beta_index, n_index, sigmanu2_index, sigmax2_index] += (ek.item() ** 2) / (number_of_repeats * number_of_last_iterations)

    return mse, msd, pup

if __name__ == '__main__':
    tau_vector = np.linspace(0.1, 2.5, 5)
    beta_vector = np.linspace(0.1, 1.0, 10)
    n_vector = [5, 10, 20]
    sigmanu2_vector = np.power(10.0, np.arange(-5, 0, 1), dtype=float)
    sigmax2_vector = np.power(10.0, np.arange(0, 3, 1), dtype=float)

    # Steady-state requires total iterations to exceed observation window
    number_of_iterations = 1500
    number_of_last_iterations = 300
    number_of_repeats = 20

    print("Generating SMNLMS dataset...")
    mse, msd, pup = evaluate_empirical_quantities_smnlms(
        tau_vector,
        beta_vector,
        n_vector,
        sigmanu2_vector,
        sigmax2_vector,
        number_of_iterations,
        number_of_last_iterations,
        number_of_repeats
    )

    data = {
        'tau': [],
        'beta': [],
        'N': [],
        'sigmanu2': [],
        'sigmax2': [],
        'MSE': [],
        'MSD': [],
        'Pup': []
    }

    for tau_index in range(len(tau_vector)):
        for beta_index in range(len(beta_vector)):
            for n_index in range(len(n_vector)):
                for sigmanu2_index in range(len(sigmanu2_vector)):
                    for sigmax2_index in range(len(sigmax2_vector)):
                        data['tau'].append(tau_vector[tau_index])
                        data['beta'].append(beta_vector[beta_index])
                        data['N'].append(n_vector[n_index])
                        data['sigmanu2'].append(sigmanu2_vector[sigmanu2_index])
                        data['sigmax2'].append(sigmax2_vector[sigmax2_index])
                        data['MSE'].append(mse[tau_index, beta_index, n_index, sigmanu2_index, sigmax2_index])
                        data['MSD'].append(msd[tau_index, beta_index, n_index, sigmanu2_index, sigmax2_index])
                        data['Pup'].append(pup[tau_index, beta_index, n_index, sigmanu2_index, sigmax2_index])

    output_path = './resultsSMNLMS.csv'
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} rows at {output_path}")