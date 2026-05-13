import time
import numpy as np
import json
from sklearn.metrics import roc_curve, roc_auc_score
from scr.qsvdd_core.engine import QuantumEngine


def test(
    n_train, X_test, Y_test, trained_params, center_train, noisy=False, ansatz="qcnn"
):
    start_time = time.time()

    y_true_local = []
    y_pred_local = []

    engine = QuantumEngine(n_qubits=4, noisy=noisy, ansatz_type=ansatz)

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


def mean_auc(
    params_list, n_train, X_test, Y_test, center_train, noisy=False, ansatz="qcnn"
):
    """
    Calcula a média e o desvio padrão do AUC repetindo o teste 5 vezes.
    O operador **kwargs captura todos os arrays e bools automaticamente.
    """
    auc_list = []
    for i in range(5):
        trained_params = params_list[i]
        auc, _, _, _, _ = test(
            n_train, X_test, Y_test, trained_params, center_train, noisy, ansatz
        )
        auc_list.append(auc)
    return np.mean(auc_list), np.std(auc_list)


def best_batch(method="qae"):
    method_upper = method.upper()

    variations = [
        "B04S2000",
        "B08S1000",
        "B16S500",
        "B32S250",
        "B64S125",
    ]

    test_results = {}

    for var in variations:
        f_name = (
            f"../results/training/{method_upper}/{method_upper}_{var}_EST_PARAMS.npy"
        )

        try:
            est_params = np.loadtxt(f_name)

            auc, y_pred, y_true, fpr, tpr = test(
                n_train, X_test, Y_test, est_params, center_train, ansatz=method
            )

            # 4. Store in the dictionary using the variation as key
            test_results[var] = {
                "auc": auc,
                "y_pred": y_pred,
                "y_true": y_true,
                "fpr": fpr,
                "tpr": tpr,
            }

            print(f"Completed: QSVDD: {method_upper} ANSATZ - {var} | AUC: {auc:.4f}")
            print("\n" + 50 * "=" + "\n")

        except FileNotFoundError:
            print(f"Error: File {f_name} not found.")
        except Exception as e:
            print(f"Error processing {var}: {e}")

    return test_results


def save_test_results(test_results, method="qae"):

    upper_method = method.upper()

    # 1. Find the configuration with the highest AUC
    best_config = max(test_results, key=lambda k: test_results[k]["auc"])
    best_data = test_results[best_config]

    print(f"The best model was: {best_config} with AUC = {best_data['auc']:.4f}")

    # 2. Save the complete dictionary (optional, in JSON format for easy reading)
    # Note: we convert numpy arrays to lists for JSON to accept
    full_results_serializable = {
        k: {
            "auc": v["auc"],
            "fpr": v["fpr"].tolist() if isinstance(v["fpr"], np.ndarray) else v["fpr"],
            "tpr": v["tpr"].tolist() if isinstance(v["tpr"], np.ndarray) else v["tpr"],
        }
        for k, v in test_results.items()
    }

    with open(f"../results/test/All_Results_{upper_method}.json", "w") as f:
        json.dump(full_results_serializable, f, indent=4)

    # 3. Save only the data of the best result for later use
    np.save(
        f"../results/test/BEST_{upper_method}_{best_config}_METRICS.npy",
        {
            "auc": best_data["auc"],
            "fpr": best_data["fpr"],
            "tpr": best_data["tpr"],
            "y_pred": best_data["y_pred"],
            "y_true": best_data["y_true"],
        },
    )

    # 4. Save a text summary for your report
    with open(f"../results/test/{upper_method}resumo_performance.txt", "w") as f:
        f.write(f"Performance Report - Ansatz: {upper_method}\n")
        f.write(f"Best Configuration: {best_config}\n")
        f.write(f"Max AUC: {best_data['auc']}\n")
        f.write("-" * 30 + "\n")
        for config, data in test_results.items():
            f.write(f"{config}: AUC = {data['auc']:.4f}\n")

    print("All files were saved successfully.")
