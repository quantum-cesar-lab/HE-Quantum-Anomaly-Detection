from qclib.state_preparation.ucge import UCGEInitialize
from qclib.state_preparation.dcsp import DcspInitialize
from qclib.state_preparation.baa_lowrank import BaaLowRankInitialize
from qclib.state_preparation.bdsp import BdspInitialize
from qclib.state_preparation.isometry import IsometryInitialize
from .channel import DepolarizingChannel_2
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit import transpile

import pennylane as qml
import numpy as np


class QCNN:
    def __init__(self, n_qubits, noisy=False):
        self.n_qubits = n_qubits
        self.noisy = noisy

        # Parâmetros reais extraídos do FakeAlgiers
        # Este setup equilibra um T1/T2 alto com uma porta CX mais lenta
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

    def _get_su_4_operator(self, params, wires):
        """
        Implementação do operador SU(4) com injeção automática de ruído.
        """

        # Bloco 1: U3 em ambos os qubits
        self._apply_gate(
            lambda: qml.U3(params[0], params[1], params[2], wires=wires[0]), [wires[0]]
        )
        self._apply_gate(
            lambda: qml.U3(params[3], params[4], params[5], wires=wires[1]), [wires[1]]
        )

        # Bloco 2: CNOT (0 -> 1)
        self._apply_gate(
            lambda: qml.CNOT(wires=[wires[0], wires[1]]),
            [wires[0], wires[1]],
            is_2q=True,
        )

        # Bloco 3: Rotações intermediárias
        self._apply_gate(lambda: qml.RY(params[6], wires=wires[0]), [wires[0]])
        self._apply_gate(lambda: qml.RZ(params[7], wires=wires[1]), [wires[1]])

        # Bloco 4: CNOT (1 -> 0)
        self._apply_gate(
            lambda: qml.CNOT(wires=[wires[1], wires[0]]),
            [wires[1], wires[0]],
            is_2q=True,
        )

        # Bloco 5: RY e Identidade (Sincronização de ruído térmico)
        self._apply_gate(lambda: qml.RY(params[8], wires=wires[0]), [wires[0]])
        # A identidade garante que o ruído térmico seja aplicado enquanto o qubit 0 opera
        self._apply_gate(lambda: qml.Identity(wires=wires[1]), [wires[1]])

        # Bloco 6: CNOT (0 -> 1)
        self._apply_gate(
            lambda: qml.CNOT(wires=[wires[0], wires[1]]),
            [wires[0], wires[1]],
            is_2q=True,
        )

        # Bloco 7: U3 finais em ambos os qubits
        self._apply_gate(
            lambda: qml.U3(params[9], params[10], params[11], wires=wires[0]),
            [wires[0]],
        )
        self._apply_gate(
            lambda: qml.U3(params[12], params[13], params[14], wires=wires[1]),
            [wires[1]],
        )

    def _get_conv_layer_1(self, params):
        # Parte 1: U nos pares (0,1) e (2,3). Qubit 4 espera.
        # O SU4 original tem várias portas. Para simplificar e manter a lógica do autor:
        self._get_su_4_operator(params, wires=[0, 1])
        self._get_su_4_operator(params, wires=[2, 3])

        if self.noisy:
            # O autor aplicava várias identidades para somar o tempo das portas do SU4
            # Vamos aplicar um erro térmico equivalente ao tempo total de um bloco SU4
            # (Aprox. 4 gates de 1q + 3 gates de 2q no seu _get_su_4_operator)
            su4_total_time = (4 * self.gate_1q) + (3 * self.gate_2q)
            qml.ThermalRelaxationError(0, self.t1, self.t2, su4_total_time, wires=4)

        qml.Barrier(wires=range(5))

        # Parte 2: U no par (4,0). Qubits 1, 2 e 3 esperam.
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

    def qcnn_ansatz_without_pooling(self, params, number_params=75):  # 75
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


class LCQHNN:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def lcqhnn_ansatz(self, params):
        """
        Implementação do Ansatz LCQHNN [1] para 5 qubits.
        [1] https://arxiv.org/pdf/2412.02059
        """
        # CORREÇÃO DO ERRO: params.shape é uma tupla, pegamos o primeiro índice

        for layer in range(self.n_qubits):
            # 1. Entranhamento Progressivo (Cadeia CNOT 0->1, 1->2...)
            # Baseado na Eq. 4 do artigo original do LCQHNN [2]
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # 2. Rotações RY Parametrizadas
            # Baseado na Eq. 5 do artigo original [2]
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)

            # 3. Entranhamento Regressivo (Cadeia CNOT invertida)
            # Assinatura 'Lean' para cancelamento de ruído e interferência
            for i in range(self.n_qubits - 1, 0, -1):
                qml.CNOT(wires=[i - 1, i])

        # 4. Transformação Final do LCQHNN
        # Aplica Hadamard no primeiro qubit para preparar a base de medição [2]
        qml.Hadamard(wires=0)

class PQC:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def pqc_ansatz(self, params):
        """
        PQC of https://arxiv.org/pdf/2308.16005
        """
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
            qml.RZ(params[i], wires=i)
            qml.RY(params[i], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CRZ(params[i], wires=[self.n_qubits - 1, i])

        qml.CRZ(params[i], wires=[self.n_qubits - 1, i])

class ProposedVQC:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

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

    def proposed_ansatz(self, params):
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

        if method == "pennylane":
            return lambda wires: qml.AmplitudeEmbedding(
                amplitude_array, wires=wires, pad_with=0.0, normalize=True
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

    # Transforma a lógica do circuito em um QNode executável
    def qc_complete_design(
        self, amplitude_array, params, method="pennylane", noisy=False
    ):
        # 1. Feature Mapping
        mapping_fn = self.feature_mapping(amplitude_array, method=method)
        mapping_fn(wires=range(self.n_qubits))

        # 2. Ansatz (QCNN com ruído interno nas portas)
        # ansatz = QCNN(self.n_qubits, noisy=noisy)
        # ansatz.qcnn_ansatz_without_pooling(params)
        ansatz = LCQHNN(self.n_qubits)
        ansatz.lcqhnn_ansatz(params)

        # # 3. Readout Error (Injetado apenas se noisy=True)
        # if noisy:
        #     readout_p = ansatz.noise_params["readout_error"]
        #     for i in range(self.n_qubits):
        #         qml.BitFlip(p=readout_p, wires=i)

        # 4. Medição
        result = (
            qml.expval(qml.PauliX(0) @ qml.PauliX(2)),
            qml.expval(qml.PauliY(0) @ qml.PauliY(2)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(2)),
        )
        return result
