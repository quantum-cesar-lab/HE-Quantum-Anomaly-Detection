# Hardware-Efficient Quantum Anomaly Detection: Navigating Architectural Trade-offs for Financial Fraud

This repository contains the implementation and experiments for the paper *"Hardware-Efficient Quantum Anomaly Detection: Navigating Architectural Trade-offs for Financial Fraud"*, benchmarking three Parameterized Quantum Circuit (PQC) architectures — **QCNN**, **QAE**, and **LCQHNN** — within the QSVDD framework for one-class classification on the Credit Card Fraud Detection dataset.

---

## Requirements

- Python 3.12.3
- [PennyLane](https://pennylane.ai/) 0.44.0

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle and place both files inside the `data/` folder:

```
data/
└── creditcard.csv
```

---

## Repository Structure

```
QSVDD2/
├── data/                         # Dataset files (not versioned)
├── notebooks/                    # Jupyter notebooks (see below)
├── results/
│   ├── plots/                    # Saved figures
│   ├── test/                     # AUC scores and ROC data from evaluation
│   └── training/                 # Loss histories, parameter histories, and timing data
├── scr/
│   └── qsvdd_core/               # Core library
│       ├── channel.py            # Custom 2-qubit depolarizing noise channel
│       ├── circuits.py           # Ansatz definitions (QCNN, QAE, LCQHNN) and feature mapping
│       ├── data_loader.py        # Data preprocessing and amplitude/angle encoding
│       └── engine.py             # QuantumEngine: device selection, QNode, and cost function
├── scripts/
│   ├── train_model.py            # circuit_training and train_five_times helpers
│   ├── test_model.py             # Evaluation: AUC scoring and result persistence
│   └── util.py                   # Timing utilities for the elapsed-time analysis
├── tests/
│   └── test_circuits.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks

The notebooks are located in the `notebooks/` folder and should be run in the following order:

### 1. `circuit_visualization.ipynb`
Draws and inspects the quantum circuits for each ansatz (QCNN, QAE, LCQHNN). Use this as a starting point to understand the circuit architecture before running any experiments.

### 2. `training_test_data.ipynb`
The main experimental notebook. Run this to reproduce all training and evaluation results. It is organized into these stages:

- **Dataset** — loads `creditcard.csv`, inspects class distribution, and prepares quantum-ready data via amplitude encoding (`QuantumDataLoader`).
- **Noiseless Training** — trains each ansatz (QCNN, QAE, LCQHNN) across the batch-size grid. Results are saved to `results/training/`.
- **Noisy Training** — repeats training with the IBM `FakeAlgiers` noise model (thermal relaxation, dephasing, gate errors, and readout errors).
- **Noiseless Test / Noisy Test** — evaluates trained models, computes AUC scores, and saves ROC data to `results/test/`.

> **Note:** Training is computationally intensive, especially for the QCNN under the noisy regime. Results are automatically persisted as `.npy` files so individual cells can be re-run without retraining.

### 3. `training_test_data_with_pca.ipynb`
An alternative training pipeline using **Angle Embedding** instead of Amplitude Embedding. Data is preprocessed with PCA (5 components) and mapped to `[0, π]` before encoding. Useful for comparing encoding strategies.

### 4. `noise_data_visualization.ipynb`
Visualizes the IBM `FakeAlgiers` noise parameters used in the simulations (T1/T2 relaxation times, gate error rates, readout errors).

### 5. `views.ipynb`
Generates all figures from the paper using the data saved in `results/`. Requires `training_test_data.ipynb` to have been run first. Sections include:

- Loss convergence curves (noiseless vs. noisy)
- Batch size impact with log-scale zoom
- Parameter evolution over training steps
- AUC distribution and cost function comparison across ansatzes
- Elapsed time regression per PQC

---

## Core Modules (`scr/qsvdd_core/`)

| File | Description |
|---|---|
| `engine.py` | `QuantumEngine` — selects `default.qubit` (noiseless) or `default.mixed` (noisy), wraps the QNode, and computes the SVDD loss |
| `circuits.py` | `QSVDDCircuit` — feature mapping strategies; `QCNNAnsatz`, `QAEAnsatz`, `LCQHNNAnsatz` — each with optional noise injection via `_apply_gate` |
| `channel.py` | `DepolarizingChannel_2` — custom 2-qubit Kraus channel used in noisy simulations |
| `data_loader.py` | `QuantumDataLoader` — MinMax scaling, zero-padding to 32 features, L2 normalization for amplitude encoding; PCA path for angle encoding |

---

## Citation

```bibtex
@article{silva2026qsvdd,
  title   = {Hardware-Efficient Quantum Anomaly Detection: Navigating Architectural Trade-offs for Financial Fraud},
  author  = {Silva, Jefferson D. S. and Gonçalves, João V. F. and Conrado, Deyvison M. and Bezerra, Pamela T. L. and Dias, Everton},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.0429000}
}
```

---

## Contact

Corresponding author: **Pamela T. L. Bezerra** — ptlb@cesar.org.br  
CESAR — Centro de Estudos e Sistemas Avançados do Recife, Recife, PE 50030-230, Brazil