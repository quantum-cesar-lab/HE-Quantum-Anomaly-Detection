from pennylane import numpy as np
from .engine import QuantumEngine
import pennylane as qml

dataset = 'fraud'
ntrain = 0 ; latent_dim = 3
num_params_conv = 45
steps = 500
learning_rate = 0.001
batch_size = 16

# 1. Inicializa a Engine globalmente (uma única vez)
engine = QuantumEngine(n_qubits=5)


def circuit_training(X_train, Y_train, n_params):
    # Inicializa parâmetros usando o numpy do PennyLane
    params = np.random.randn(n_params, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    param_history = [params]
    loss_history = []

    for it in range(steps):
        # Seleção do batch
        batch_index = np.random.randint(0, len(X_train), (batch_size,))

        # Importante: converter para array do PennyLane para manter diferenciação
        X_batch = np.array(X_train[batch_index], requires_grad=False)
        Y_batch = np.array(Y_train[batch_index], requires_grad=False)

        # 2. Chama o método cost da engine através do otimizador
        # O lambda v passa os parâmetros que o Adam está tentando ajustar
        params, cost_new = opt.step_and_cost(
            lambda v: engine.cost(v, X_batch, Y_batch),
            params
        )

        param_history.append(params)
        loss_history.append(cost_new)


        print(f"iteration: {it} | cost: {cost_new:.6f}")

    return loss_history, params, param_history
