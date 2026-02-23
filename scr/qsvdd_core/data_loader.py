import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class QuantumDataLoader:
    def __init__(self):
        self.n_qubits = 5  # Defined for 32 amplitudes (2^5)
        self.target_dim = 2**self.n_qubits
        self.scaler = MinMaxScaler()

    def prepare_fraud_data(self, df):

        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        X_scaled = self.scaler.fit_transform(X)

        n_samples, n_features = X_scaled.shape
        padding_size = self.target_dim - n_features
        X_padded = np.pad(
            X_scaled, ((0, 0), (0, padding_size)), mode="constant", constant_values=0
        )

        norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
        # Avoids division by zero
        norms[norms == 0] = 1.0
        X_quantum = X_padded / norms

        return X_quantum, y