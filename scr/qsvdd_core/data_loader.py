import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


class QuantumDataLoader:
    def __init__(self):
        self.n_qubits = 5  # Definido para 32 amplitudes (2^5)
        self.target_dim = 2**self.n_qubits
        self.scaler = MinMaxScaler()

        self.std_scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_qubits)
        self.angle_map_scaler = MinMaxScaler(feature_range=(0, np.pi))

    def prepare_fraud_data(self, df):
        """
        Processa o dataset de fraude específico do arquivo carregado.
        """

        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        X_scaled = self.scaler.fit_transform(X)

        n_samples, n_features = X_scaled.shape
        padding_size = self.target_dim - n_features
        X_padded = np.pad(
            X_scaled, ((0, 0), (0, padding_size)), mode="constant", constant_values=0
        )

        norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
        # Evita divisão por zero
        norms[norms == 0] = 1.0
        X_quantum = X_padded / norms

        return X_quantum, y

    def prepare_pca_data(self, df):
        """Novo Caminho: Focado em Angle Embedding"""
        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        # Fluxo: Padronizar -> PCA -> Mapear para [0, pi]
        X_std = self.std_scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        print('Explained Variance Ratio: ', self.pca.explained_variance_ratio_)
        print('Singular Values: ', self.pca.singular_values_)
        X_angles = self.angle_map_scaler.fit_transform(X_pca)

        return X_angles, y

    def plot_correlation_matrix(self, df, scale=0.7):
        """Mostra a correlação apenas entre as features (X)"""
        # Dropar a coluna de rótulo para focar apenas nas características
        df_features = df.drop("Class", axis=1)
        df_features_std = self.std_scaler.fit_transform(df_features)
        df_scaled = pd.DataFrame(df_features_std, columns=df_features.columns)

        sns.set(font_scale=scale)
        plt.figure(figsize=(15*scale, 12*scale))
        corr = df_scaled.corr()

        # Usando uma máscara para mostrar apenas a parte de baixo do triângulo (opcional, mas fica mais limpo)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})

        plt.title("Matriz de Correlação das Features (Sem o Rótulo)")
        plt.show()

    def plot_explained_variance(self, df):
        """Gera o gráfico de cotovelo para justificar o número de qubits"""
        X = df.drop("Class", axis=1).values
        X_std = self.std_scaler.fit_transform(X)

        # Calculamos o PCA para todas as componentes possíveis (30)
        pca_full = PCA().fit(X_std)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 marker='o', linestyle='--', color='b')

        # Destaca o ponto atual de 5 qubits
        plt.axvline(x=self.n_qubits, color='r', linestyle=':', label=f'Atual: {self.n_qubits} Qubits')
        plt.axhline(y=cumulative_variance[self.n_qubits - 1], color='r', linestyle=':')

        plt.xlabel('Número de Componentes Principais (Qubits)')
        plt.ylabel('Variância Explicada Acumulada')
        plt.title('Análise de Dimensionalidade para o QSVDD')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        print(f"Variância com {self.n_qubits} qubits: {cumulative_variance[self.n_qubits - 1]:.2%}")
