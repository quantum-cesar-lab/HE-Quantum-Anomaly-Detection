from .circuits import QSVDDCircuit
import pennylane as qml
from pennylane import numpy as np


class QuantumEngine:
    def __init__(self, n_qubits, noisy=False, ansatz_type="qcnn"):
        self.n_qubits = n_qubits
        self.noisy = noisy
        self.ansatz_type = ansatz_type

        # Seleção dinâmica do dispositivo
        if self.noisy:
            # Necessário para suportar ThermalRelaxation e DepolarizingChannel_2
            self.dev = qml.device("default.mixed", wires=self.n_qubits)
        else:
            # Muito mais rápido para simulações de estado puro
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.circuit_logic = QSVDDCircuit(self.n_qubits)
        self.quantum_circuit = qml.qnode(self.dev)(self._circuit_definition)

    def _circuit_definition(self, x, params):
        """Esta é a função interna que define as portas quânticas."""
        return self.circuit_logic.qc_complete_design(
            x, params, noisy=self.noisy, ansatz_type=self.ansatz_type
        )

    def cost(self, params, X, Y):
        # np.stack do PennyLane preserva a diferenciabilidade
        predictions = np.stack([self.quantum_circuit(x, params) for x in X])
        loss_value = np.mean((predictions - Y) ** 2)
        return loss_value
