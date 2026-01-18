import numpy as np
from qiskit_algorithms.optimizers import SPSA  # Otimizador padrão para VQC
from .engine import QuantumEngine
from .circuits import QSVDDCircuit


def cost_function(params, X_batch, Y_batch, circuit_design, engine):
    """
    Calcula o MSE entre as predições do circuito e o centro (Y_batch).
    """
    predictions = []
    for x in X_batch:
        # 1. Gera o circuito com os parâmetros atuais
        qc = circuit_design.qc_complete_design(x, params, method="qiskit")

        # 2. Executa no simulador e pega as correlações
        pred = engine.get_expectation_value(qc, params)
        predictions.append(pred)

    predictions = np.array(predictions)
    # MSE: (Predição - Centro)^2
    loss = np.mean(np.sum(np.square(predictions - Y_batch), axis=1))
    return loss


def circuit_training(X_train, Y_train, num_params, steps=50):
    # Inicialização de componentes
    engine = QuantumEngine()
    q_circuit = QSVDDCircuit(n_qubits=5)

    # Inicializa parâmetros aleatórios (NumPy puro)
    params = np.random.randn(num_params)

    loss_history = []
    param_history = [params]

    optimizer = SPSA(maxiter=1)

    print("Iniciando treinamento QSVDD...")

    for it in range(steps):

        # Seleção do Batch
        batch_index = np.random.randint(0, len(X_train), size=2)
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]

        def objective(p):
            return cost_function(p, X_batch, Y_batch, q_circuit, engine)

        result = optimizer.minimize(fun=objective, x0=params)


        params = result.x
        cost_new = result.fun

        param_history.append(params)
        loss_history.append(cost_new)

        print(f"iteration: {it} | cost: {cost_new}")

    return loss_history, params, param_history