from qclib.state_preparation.ucge import UCGEInitialize
from qclib.state_preparation.dcsp import DcspInitialize
from qclib.state_preparation.baa_lowrank import BaaLowRankInitialize
from qclib.state_preparation.bdsp import BdspInitialize
from qclib.state_preparation.isometry import IsometryInitialize
from .channel import DepolarizingChannel_2
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit import transpile
from itertools import combinations

import pennylane as qml
import numpy as np


class BaseAnsatz:
    def __init__(self, n_qubits, noisy=False):
        self.n_qubits = n_qubits
        self.noisy = noisy

        # Parâmetros reais extraídos do FakeAlgiers
        # Este setup equilibra um T1/T2 alto com uma porta CX mais lenta em relaçao ao torino
        self.noise_params = {
            "p": 0.006471,  # Erro de despolarização médio
            "t1": 156847.12,  # T1 médio (~156.8 microssegundos)
            "t2": 188818.14,  # T2 médio (~188.8 microssegundos)
            "gate_1q": 35.56,  # Duração da porta single-qubit (ns)
            "gate_2q": 259.56,  # Duração da porta CX (ns)
            "readout_error": 0.00744,  # Erro de medição (~0.7%)
        }

        # Facilitadores de acesso direto
        self.p = self.noise_params["p"]
        self.t1 = self.noise_params["t1"]
        self.t2 = self.noise_params["t2"]
        self.gate_1q = self.noise_params["gate_1q"]
        self.gate_2q = self.noise_params["gate_2q"]

    def _apply_gate(self, gate_fn, wires, is_2q=False):
        """Aplica a porta e insere ruído se noisy=True."""
        gate_fn()
        if self.noisy:
            t_gate = self.gate_2q if is_2q else self.gate_1q

            # 1. Erro de Relaxamento Térmico (T1 e T2)
            for wire in wires:
                qml.ThermalRelaxationError(0, self.t1, self.t2, t_gate, wires=wire)

            # 2. Erro de Despolarização (Apenas para portas de 2 qubits)
            if is_2q:

                kraus_ops = DepolarizingChannel_2.compute_kraus_matrices(self.p)

                # O QubitChannel "embrulha" essas matrizes para o simulador
                qml.QubitChannel(kraus_ops, wires=wires)


class QCNNAnsatz(BaseAnsatz):

    def _get_su_4_operator(self, params, wires):
        """
        Implementação do operador SU(4) com injeção automática de ruído.
        """

        self._apply_gate(
            lambda: qml.U3(params[0], params[1], params[2], wires=wires[0]), [wires[0]]
        )
        self._apply_gate(
            lambda: qml.U3(params[3], params[4], params[5], wires=wires[1]), [wires[1]]
        )

        self._apply_gate(
            lambda: qml.CNOT(wires=[wires[0], wires[1]]),
            [wires[0], wires[1]],
            is_2q=True,
        )

        self._apply_gate(lambda: qml.RY(params[6], wires=wires[0]), [wires[0]])
        self._apply_gate(lambda: qml.RZ(params[7], wires=wires[1]), [wires[1]])

        self._apply_gate(
            lambda: qml.CNOT(wires=[wires[1], wires[0]]),
            [wires[1], wires[0]],
            is_2q=True,
        )

        self._apply_gate(lambda: qml.RY(params[8], wires=wires[0]), [wires[0]])
        self._apply_gate(lambda: qml.Identity(wires=wires[1]), [wires[1]])

        self._apply_gate(
            lambda: qml.CNOT(wires=[wires[0], wires[1]]),
            [wires[0], wires[1]],
            is_2q=True,
        )

        self._apply_gate(
            lambda: qml.U3(params[9], params[10], params[11], wires=wires[0]),
            [wires[0]],
        )
        self._apply_gate(
            lambda: qml.U3(params[12], params[13], params[14], wires=wires[1]),
            [wires[1]],
        )

    def _get_conv_layer_1(self, params):

        self._get_su_4_operator(params, wires=[0, 1])
        self._get_su_4_operator(params, wires=[2, 3])

        if self.noisy:
            su4_total_time = (4 * self.gate_1q) + (3 * self.gate_2q)
            qml.ThermalRelaxationError(0, self.t1, self.t2, su4_total_time, wires=4)

        qml.Barrier(wires=range(5))

        self._get_su_4_operator(params, wires=[4, 0])
        if self.noisy:
            for i in [1, 2, 3]:
                qml.ThermalRelaxationError(0, self.t1, self.t2, su4_total_time, wires=i)

        qml.Barrier(wires=range(5))

        # Parte 3: U nos pares (1,2) e (3,4). Qubit 0 espera.
        self._get_su_4_operator(params, wires=[1, 2])
        self._get_su_4_operator(params, wires=[3, 4])
        if self.noisy:
            qml.ThermalRelaxationError(0, self.t1, self.t2, su4_total_time, wires=0)

    def _get_conv_layer_2(self, params):
        self._get_su_4_operator(params, wires=[0, 1])
        self._get_su_4_operator(params, wires=[2, 3])
        self._get_su_4_operator(params, wires=[0, 3])
        self._get_su_4_operator(params, wires=[1, 2])

    def _get_conv_layer_3(self, params):
        self._get_su_4_operator(params, wires=[0, 1])

    def build(self, params, number_params=75):  # 75
        param1 = params[0:number_params]
        param2 = params[number_params : 2 * number_params]
        param3 = params[2 * number_params : 3 * number_params]
        param4 = params[3 * number_params : 4 * number_params]
        param5 = params[4 * number_params : 5 * number_params]

        self._get_conv_layer_1(param1)
        self._get_conv_layer_1(param2)
        self._get_conv_layer_2(param3)
        self._get_conv_layer_2(param4)
        self._get_conv_layer_3(param5)


class QAEAnsatz(BaseAnsatz):

    def _get_u_operator_qae(self, params):
        nqubits = 5
        ntrash = 3

        # 1. Rotações iniciais
        for i in range(nqubits):
            self._apply_gate(
                lambda i=i, p=params[i]: qml.RY(p, wires=i),  # Captura i e p
                wires=[i],
                is_2q=False,
            )

        # 2. CZs entre trash qubits
        for i, j in combinations(range(0, ntrash), 2):
            self._apply_gate(
                lambda i=i, j=j: qml.CZ(wires=[i, j]),  # Captura i e j
                wires=[i, j],
                is_2q=True,
            )

        # 3. CZs entre trash e non-trash
        for idx in range(ntrash):
            for i in range(ntrash):
                for j in range(ntrash + i, nqubits, ntrash):
                    w1 = (idx + i) % (ntrash)
                    w2 = j
                    self._apply_gate(
                        lambda w1=w1, w2=w2: qml.CZ(wires=[w1, w2]),
                        wires=[w1, w2],
                        is_2q=True,
                    )

    def _get_u_qae_last(self, params):
        ntrash = 3
        for i in range(ntrash):
            self._apply_gate(
                lambda i=i, p=params[i]: qml.RY(p, wires=i),  # Captura i e p
                wires=[i],
                is_2q=False,
            )

    def build(self, params, number_params=78):
        """
        Build dinâmico para o QAE.
        Total de parâmetros necessários: (9 blocos * 5) + 3 = 48.
        """
        n_blocos = 9
        params_per_block = 5

        # 1. Aplica os 9 blocos de codificação
        for i in range(n_blocos):
            start = i * params_per_block
            end = start + params_per_block
            # Passa apenas o pedaço de 5 parâmetros para cada bloco
            self._get_u_operator_qae(params[start:end])

        # 2. Aplica o bloco final (usando os parâmetros a partir do índice 45)
        # params[45:48] pegará os 3 últimos valores se o total for 48
        self._get_u_qae_last(params[45:48])


class LCQHNNAnsatz(BaseAnsatz):

    def build(self, params):
        """
        Implementação do Ansatz LCQHNN [1] para 5 qubits.
        [1] https://arxiv.org/pdf/2412.02059
        """

        for layer in range(1): # self.n_qubits

            for i in range(self.n_qubits - 1):
                self._apply_gate(
                    lambda i=i: qml.CNOT(wires=[i, i + 1]), wires=[i, i + 1], is_2q=True
                )

            for i in range(self.n_qubits):
                self._apply_gate(
                    lambda i=i, p=params[i]: qml.RY(p, wires=i), wires=[i], is_2q=False
                )

            for i in range(self.n_qubits - 1, 0, -1):
                self._apply_gate(
                    lambda i=i: qml.CNOT(wires=[i - 1, i]), wires=[i - 1, i], is_2q=True
                )

        self._apply_gate(lambda: qml.Hadamard(wires=0), wires=[0], is_2q=False)


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

    def feature_mapping(self, amplitude_array, method="baa_lowrank"):
        """
        Initializes quantum state using various state preparation methods.
        """
        if hasattr(amplitude_array, "numpy"):
            amplitude_array = amplitude_array.numpy()
        else:
            amplitude_array = np.array(amplitude_array, dtype=float)

        if method == "pennylane":
            return lambda wires: qml.AmplitudeEmbedding(
                amplitude_array, wires=wires, pad_with=0.0, normalize=True
            )

        if method == "angle_embedding":
            return lambda wires: qml.AngleEmbedding(
                amplitude_array, wires=wires, rotation="X"
            )


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

    def qc_complete_design(
        self,
        amplitude_array,
        params,
        f_method="pennylane",
        noisy=False,
        ansatz_type="lcqhnn",
    ):

        mapping_fn = self.feature_mapping(amplitude_array, method=f_method)
        mapping_fn(wires=range(self.n_qubits))

        ansatz_classes = {
            "qcnn": QCNNAnsatz,
            "lcqhnn": LCQHNNAnsatz,
            "qae": QAEAnsatz,
        }

        if ansatz_type not in ansatz_classes:
            raise ValueError(f"Ansatz '{ansatz_type}' não reconhecido.")

        ansatz_obj = ansatz_classes[ansatz_type](self.n_qubits, noisy=noisy)
        ansatz_obj.build(params)

        if noisy and hasattr(ansatz_obj, "noise_params"):
            readout_p = ansatz_obj.noise_params["readout_error"]
            for i in range(self.n_qubits):
                qml.BitFlip(p=readout_p, wires=i)

        if ansatz_type == "lcqhnn":
            # Como o Hadamard já está no Ansatz, medimos todos em Pauli-Z
            result = tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits))
        else:
            # Padrão original para os outros métodos
            result = (
                qml.expval(qml.PauliX(0) @ qml.PauliX(2)),
                qml.expval(qml.PauliY(0) @ qml.PauliY(2)),
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(2)),
            )
        return result

    # class ProposedVQC:
    #     def __init__(self, n_qubits):
    #         self.n_qubits = n_qubits
    #
    #     def _get_rx_layer(self, param_pack):
    #         n = self.n_qubits
    #         for i in range(n):
    #             qml.RX(param_pack[i], wires=i)
    #
    #     def _get_cyclic_ansatz_layer(self, param_pack):
    #
    #         n = self.n_qubits
    #
    #         for i in range(0, n - 1, 2):
    #             p = param_pack[i * 3 : (i + 1) * 3]
    #             qml.CRot(p[0], p[1], p[2], wires=[i, i + 1])
    #
    #         for i in range(1, n - 1, 2):
    #             p = param_pack[i * 3 : (i + 1) * 3]
    #             qml.CRot(p[0], p[1], p[2], wires=[i, i + 1])
    #
    #         if n > 1:
    #             p = param_pack[(n - 1) * 3 : n * 3]
    #             qml.CRot(p[0], p[1], p[2], wires=[n - 1, 0])
    #
    #     def build(self, params):
    #         n = self.n_qubits
    #
    #         param_pack1 = params[:n]
    #         param_pack2 = params[n : 4 * n]
    #         param_pack3 = params[4 * n : 5 * n]
    #         param_pack4 = params[5 * n : 8 * n]
    #         param_pack5 = params[8 * n :]  # 9*n parameters
    #
    #         self._get_rx_layer(param_pack1)
    #
    #         self._get_cyclic_ansatz_layer(param_pack2)
    #
    #         self._get_rx_layer(param_pack3)
    #
    #         self._get_cyclic_ansatz_layer(param_pack4)
    #
    #         self._get_rx_layer(param_pack5)
