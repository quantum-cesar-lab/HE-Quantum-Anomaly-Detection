import numpy as np
from .circuits import QSVDDCircuit
import pennylane as qml

dataset = 'fraud'
ntrain = 0 ; latent_dim = 3
num_params_conv = 45
steps = 500
learning_rate = 0.001
batch_size = 16

def svdd_loss(Y, predictions):
    loss = 0
    for l, p in zip(Y, predictions):
        loss = loss + np.sum(np.array(p-l)**2)
    loss = loss / len(Y)
    return loss


def cost(params, X, Y):
    predictions = np.array([QSVDDCircuit.qc_complete_design(x, params) for x in X])
    loss_value = np.square(np.subtract(predictions, Y)).mean()
    return loss_value


def circuit_training(X_train, Y_train, n_params):

    params = np.random.randn(n_params, requires_grad = True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    param_history= [params]
    loss_history = []

    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = np.array([X_train[i] for i in batch_index])
        Y_batch = np.array([Y_train[i] for i in batch_index])

        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch), params)
        param_history.append(params)
        loss_history.append(cost_new)


        print("iteration: ", it, " cost: ", cost_new)


    return loss_history, params, param_history
