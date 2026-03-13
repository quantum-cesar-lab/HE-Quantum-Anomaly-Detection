import numpy as np
import os


def get_time_elapsed(ansatz='qcnn'):
    ansatz = ansatz.upper()
    mean_time = []
    std_time = []

    # Range 100 a 1000 (inclusive) de 50 em 50
    for step in range(100, 1050, 50):

        f_name = f'../results/training/{ansatz}/{ansatz}_B04S{step}_TIME_MEAN.npy'

        if os.path.exists(f_name):
            data = np.loadtxt(f_name)
            mean_time.append(np.mean(data))
            std_time.append(np.std(data))
        else:
            print(f"Aviso: Arquivo {f_name} não encontrado.")

    return np.array(mean_time), np.array(std_time)


def get_time_elapsed2(ansatz='qcnn'):
    ansatz = ansatz.upper()
    mean_time = []

    # Range 100 a 1000 (inclusive) de 50 em 50
    for step in range(50, 500, 50):
        print(step)

        f_name = f'../results/training/{ansatz}/{ansatz}_B04S{step}_TIME_SINGLE_EXEC.npy'

        if os.path.exists(f_name):
            data = np.loadtxt(f_name)
            mean_time.append(data)
        else:
            print(f"Aviso: Arquivo {f_name} não encontrado.")

    return np.array(mean_time)