import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


class QuantumDataLoader:
    def __init__(self):
        self.n_qubits = 5  # Defined for 32 amplitudes (2^5)
        self.target_dim = 2 ** self.n_qubits
        self.scaler = MinMaxScaler()

        self.std_scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_qubits)
        self.angle_map_scaler = MinMaxScaler(feature_range=(0, np.pi))

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

    def prepare_bc_data(self, x_data):
        X_scaled = self.scaler.fit_transform(x_data)

        n_samples, n_features = X_scaled.shape
        padding_size = self.target_dim - n_features
        X_padded = np.pad(
            X_scaled, ((0, 0), (0, padding_size)), mode="constant", constant_values=0
        )

        norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
        # Avoids division by zero
        norms[norms == 0] = 1.0
        X_quantum = X_padded / norms

        return X_quantum

    def prepare_census_data(self, x_data):
        X_scaled = self.scaler.fit_transform(x_data)

        n_samples, n_features = X_scaled.shape
        padding_size = self.target_dim - n_features
        X_padded = np.pad(
            X_scaled, ((0, 0), (0, padding_size)), mode="constant", constant_values=0
        )

        norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
        # Avoids division by zero
        norms[norms == 0] = 1.0
        X_quantum = X_padded / norms

        return X_quantum

    def prepare_classic_data(self, df):
        # Flow for classic algorithms (IF, LOF, SVM, Deep-SVDD)
        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        X_classic = self.std_scaler.fit_transform(X)

        return X_classic, y

    def prepare_pca_data(self, df):
        """New Path: Focused on Angle Embedding"""
        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        X_std = self.std_scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        print("Explained Variance Ratio: ", self.pca.explained_variance_ratio_)
        print("Singular Values: ", self.pca.singular_values_)
        X_angles = self.angle_map_scaler.fit_transform(X_pca)

        return X_angles, y

    def plot_correlation_matrix(self, df, scale=0.7):
        """Shows the correlation only between the features (X)"""
        df_features = df.drop("Class", axis=1)
        df_features_std = self.std_scaler.fit_transform(df_features)
        df_scaled = pd.DataFrame(df_features_std, columns=df_features.columns)

        sns.set(font_scale=scale)
        plt.figure(figsize=(15 * scale, 12 * scale))
        corr = df_scaled.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Feature Correlation Matrix (Without Label)")
        plt.show()

    def plot_explained_variance(self, df):
        """Generates the elbow plot to justify the number of qubits"""
        X = df.drop("Class", axis=1).values
        X_std = self.std_scaler.fit_transform(X)

        pca_full = PCA().fit(X_std)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o",
            linestyle="--",
            color="b",
        )

        plt.axvline(
            x=self.n_qubits,
            color="r",
            linestyle=":",
            label=f"Current: {self.n_qubits} Qubits",
        )
        plt.axhline(y=cumulative_variance[self.n_qubits - 1], color="r", linestyle=":")

        plt.xlabel("Number of Principal Components (Qubits)")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Dimensionality Analysis for QSVDD")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        print(
            f"Variance with {self.n_qubits} qubits: {cumulative_variance[self.n_qubits - 1]:.2%}"
        )