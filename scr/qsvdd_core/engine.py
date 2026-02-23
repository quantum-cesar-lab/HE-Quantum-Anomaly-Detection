from .circuits import QSVDDCircuit
import pennylane as qml
from pennylane import numpy as np


class QuantumEngine:
    def __init__(self, n_qubits, noisy=False, fm="pennylane", ansatz_type="qcnn"):
        self.n_qubits = n_qubits
        self.noisy = noisy
        self.ansatz_type = ansatz_type
        self.fm = fm

        # Dynamic device selection
        if self.noisy:
            # Necessary to support ThermalRelaxation and DepolarizingChannel_2
            self.dev = qml.device("default.mixed", wires=self.n_qubits)
        else:
            # Much faster for pure state simulations
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.circuit_logic = QSVDDCircuit(self.n_qubits)
        self.quantum_circuit = qml.qnode(self.dev)(self._circuit_definition)

    def _circuit_definition(self, x, params):
        """Internal function that defines the quantum gates."""
        return self.circuit_logic.qc_complete_design(
            x, params, f_method=self.fm, noisy=self.noisy, ansatz_type=self.ansatz_type
        )

    def cost(self, params, X, Y):
        # PennyLane's np.stack preserves differentiability
        predictions = np.stack([self.quantum_circuit(x, params) for x in X])
        loss_value = np.mean((predictions - Y) ** 2)
        return loss_value