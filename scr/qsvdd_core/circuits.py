from qclib.state_preparation.ucge import UCGEInitialize
from qclib.state_preparation.dcsp import DcspInitialize
from qclib.state_preparation.baa_lowrank import BaaLowRankInitialize
from qclib.state_preparation.bdsp import BdspInitialize
from qclib.state_preparation.isometry import IsometryInitialize

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit import transpile

import pennylane as qml
import numpy as np


class QSVDDCircuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def _get_empty_circuit(self):
        return QuantumCircuit(self.n_qubits)

    def _get_qiskit_state_prep(self, amplitude_array):
        qc = self._get_empty_circuit()
        gate = StatePreparation(amplitude_array)
        qc.append(gate, range(self.n_qubits))
        return qc

    def _get_rx_layer(self, param_pack):
        n = self.n_qubits
        for i in range(n):
            qml.RX(param_pack[i], wires=i)

    def _get_cyclic_ansatz_layer(self, param_pack):

        n = self.n_qubits

        for i in range(0, n - 1, 2):
            p = param_pack[i * 3 : (i + 1) * 3]
            qml.CRot(p[0], p[1], p[2], wires=[i, i + 1])

        for i in range(1, n - 1, 2):
            p = param_pack[i * 3 : (i + 1) * 3]
            qml.CRot(p[0], p[1], p[2], wires=[i, i + 1])

        if n > 1:
            p = param_pack[(n - 1) * 3 : n * 3]
            qml.CRot(p[0], p[1], p[2], wires=[n - 1, 0])

    def ansatz(self, params):

        n = self.n_qubits

        param_pack1 = params[:n]
        param_pack2 = params[n : 4 * n]
        param_pack3 = params[4 * n : 5 * n]
        param_pack4 = params[5 * n : 8 * n]
        param_pack5 = params[8 * n :]  # 9*n parameters

        self._get_rx_layer(param_pack1)

        self._get_cyclic_ansatz_layer(param_pack2)

        self._get_rx_layer(param_pack3)

        self._get_cyclic_ansatz_layer(param_pack4)

        self._get_rx_layer(param_pack5)

    def feature_mapping(self, amplitude_array, method="baa_lowrank"):
        """
        Initializes quantum state using various state preparation methods.
        """
        if hasattr(amplitude_array, "numpy"):
            amplitude_array = amplitude_array.numpy()

        norm = np.linalg.norm(amplitude_array)
        if norm > 0:
            amplitude_array = amplitude_array / norm

        initializers = {
            "ucge": lambda: UCGEInitialize(amplitude_array).definition,
            "dcsp": lambda: DcspInitialize(amplitude_array).definition,
            "baa_lowrank": lambda: BaaLowRankInitialize(amplitude_array).definition,
            "bdsp": lambda: BdspInitialize(amplitude_array).definition,
            "isometry": lambda: IsometryInitialize(amplitude_array).definition,
            "qiskit": lambda: self._get_qiskit_state_prep(amplitude_array),
        }

        if method not in initializers:
            raise ValueError(
                f"Method '{method}' not recognized. Available: {list(initializers.keys())}"
            )
        qc_decomposed = initializers[method]().decompose().decompose()
        transpiled_fm = transpile(qc_decomposed, basis_gates=["u", "cx"])

        return qml.from_qiskit(transpiled_fm)

    # Transforma a lógica do circuito em um QNode executável
    def qc_complete_design(self, amplitude_array, params, method="baa_lowrank"):

        # mapping_fn = self.feature_mapping(amplitude_array, method=method)
        # mapping_fn(wires=range(self.n_qubits))
        qml.AmplitudeEmbedding(amplitude_array, wires=range(self.n_qubits), pad_with=0., normalize=True)

        self.ansatz(params)

        result = (
            qml.expval(qml.PauliX(0) @ qml.PauliX(2)),
            qml.expval(qml.PauliY(0) @ qml.PauliY(2)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(2)),
        )
        return result
