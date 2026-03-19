import time
from pennylane import numpy as np

from scr.qsvdd_core.engine import QuantumEngine
import pennylane as qml


def circuit_training(
    X_train,
    Y_train,
    batch_size,
    learning_rate,
    steps,
    noisy=False,
    ansatz="qcnn",
    seed=42,
    fm="pennylane",
):
    engine = QuantumEngine(n_qubits=5, noisy=noisy, fm=fm, ansatz_type=ansatz)

    params_map = {"qcnn": 375, "lcqhnn": 5, "qae": 48}
    n_params = params_map[ansatz]

    np.random.seed(seed)
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

        #print(f"iteration: {it} | cost: {cost_new:.6f}")

    return loss_history, params, param_history


def train_five_times(**kwargs):
    est_params_list = []
    loss_history_list = []
    param_history_list = []
    time_records = []
    rng = np.random.default_rng()
    for i in range(5):
        current_args = kwargs.copy()
        seed = rng.integers(low=0, high=2**32)
        current_args["seed"] = seed
        print(f"--- Starting training round {i + 1} with seed {seed} ---")
        time_start = time.time()
        loss_history, trained_params, param_history = circuit_training(**current_args)
        time_records.append(time.time() - time_start)
        est_params_list.append(trained_params)
        loss_history_list.append(loss_history)
        param_history_list.append(param_history)

    return loss_history_list, est_params_list, param_history_list, time_records
