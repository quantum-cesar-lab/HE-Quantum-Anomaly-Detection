import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scr.qsvdd_core.engine import QuantumEngine


def test(n_train, X_test, Y_test, trained_params, center_train):
    start_time = time.time()

    y_true_local = []
    y_pred_local = []

    engine = QuantumEngine(5)

    for i in range(2):
        step_start_time = time.time()

        filter_idx = np.where(Y_test == i)[0]
        test_data = X_test[filter_idx]

        label = 0 if i == n_train else 1
        print(f"Processing class {i} (label {label}) | Samples: {len(test_data)}")

        pred = np.asarray(
            [engine.quantum_circuit(x, trained_params) for x in test_data]
        )

        for j in range(len(pred)):

            diff = pred[j] - center_train
            dist = np.mean(diff**2)

            y_pred_local.append(dist)
            y_true_local.append(label)

        print(f"Finished class {i} in {time.time() - step_start_time:.2f}s")

    auc = roc_auc_score(y_true_local, y_pred_local)
    fpr, tpr, thresholds = roc_curve(y_true_local, y_pred_local)

    total_time = time.time() - start_time
    print(f"Test completed in {total_time:.2f}s | AUC: {auc:.4f}")

    return auc, y_pred_local, y_true_local, fpr, tpr
