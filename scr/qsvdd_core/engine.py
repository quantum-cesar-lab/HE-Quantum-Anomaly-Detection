from .circuits import QSVDDCircuit
import pennylane as qml
from pennylane import numpy as np


class QuantumEngine:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.circuit_logic = QSVDDCircuit(self.n_qubits)
        self.quantum_circuit = qml.qnode(self.dev)(self._circuit_definition)

    def _circuit_definition(self, x, params):
        """Esta é a função interna que define as portas quânticas."""
        return self.circuit_logic.qc_complete_design(x, params)

    def cost(self, params, X, Y):
        # np.stack do PennyLane preserva a diferenciabilidade
        predictions = np.stack([self.quantum_circuit(x, params) for x in X])
        loss_value = np.mean((predictions - Y) ** 2)
        return loss_value
