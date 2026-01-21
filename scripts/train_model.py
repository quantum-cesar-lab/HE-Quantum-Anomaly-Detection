from pennylane import numpy as np
from scr.qsvdd_core.engine import QuantumEngine
import pennylane as qml


def circuit_training(X_train, Y_train, n_params, batch_size, learning_rate, steps):
    engine = QuantumEngine(n_qubits=5)

    params = np.random.randn(n_params, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    param_history = [params]
    loss_history = []

    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))

        X_batch = np.array(X_train[batch_index], requires_grad=False)
        Y_batch = np.array(Y_train[batch_index], requires_grad=False)

        params, cost_new = opt.step_and_cost(
            lambda v: engine.cost(v, X_batch, Y_batch), params
        )

        param_history.append(params)
        loss_history.append(cost_new)

        print(f"iteration: {it} | cost: {cost_new:.6f}")

    return loss_history, params, param_history
