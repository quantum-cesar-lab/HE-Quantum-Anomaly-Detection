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
):
    engine = QuantumEngine(n_qubits=5, noisy=noisy, ansatz_type=ansatz)

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

        print(f"iteration: {it} | cost: {cost_new:.6f}")

    return loss_history, params, param_history


def circuit_training2(
    X_train,
    Y_train,
    batch_size,
    learning_rate,
    steps,
    noisy=False,
    fm="pennylane",
    ansatz="qcnn",
):
    engine = QuantumEngine(n_qubits=5, noisy=noisy, fm=fm, ansatz_type=ansatz)

    params_map = {"qcnn": 375, "lcqhnn": 5, "qae": 48}
    n_params = params_map[ansatz]

    np.random.seed(42)
    params = np.random.randn(n_params, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    param_history = [params]
    loss_history = []

    it = 0
    while it < steps:
        success = False
        attempts = 0
        max_attempts = (
            10  # Aumentamos para garantir resiliência contra divergências numéricas
        )

        while not success and attempts < max_attempts:
            try:
                # 1. Seleção do batch
                batch_index = np.random.randint(0, len(X_train), (batch_size,))
                X_batch = np.array(X_train[batch_index], requires_grad=False)
                Y_batch = np.array(Y_train[batch_index], requires_grad=False)

                params, cost_new = opt.step_and_cost(
                    lambda v: engine.cost(v, X_batch, Y_batch), params
                )

                success = True  # Otimização bem-sucedida

            except Exception as e:
                attempts += 1
                print(
                    f"⚠️ Iteração {it}: Falha numérica (Tentativa {attempts}/{max_attempts}). Erro: {e}"
                )

        if success:
            param_history.append(params)
            loss_history.append(cost_new)
            print(f"iteration: {it} | cost: {cost_new:.6f}")
            it += 1
        else:
            print(
                f"❌ Erro crítico: Não foi possível encontrar um batch estável para a iteração {it}."
            )
            break

    return loss_history, params, param_history


def train_five_times(**kwargs):
    params_list = []
    for i in range(5):
        current_args = kwargs.copy()
        current_args['seed'] = i
        print(f"--- Starting training round {i + 1} with seed {i} ---")
        _, trained_params, _ = circuit_training(**current_args)
        params_list.append(trained_params)

    return params_list