from qclib.state_preparation.ucge import UCGEInitialize
from qclib.state_preparation.dcsp import DcspInitialize
from qiskit import QuantumCircuit
from qiskit.circuit.library import CU3Gate, StatePreparation



class QSVDDCircuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits


    def _get_empty_circuit(self):
        return QuantumCircuit(self.n_qubits)


    def feature_mapping(self, amplitude_array, method='ucge'):
        """
        Initializes quantum state using the UCGE method[1].

        This technique optimizes state preparation by detecting unentangled
        substates and simplifying quantum multiplexers to reduce circuit
        depth and CNOT gate count[1].

        Ref: [1] Carvalho et al. (2025). https://doi.org/10.1145/3748260

        Args:
            amplitude_array (np.ndarray): Normalized amplitude vector of size 2^n[cite: 68].
        """
        qc = self._get_empty_circuit()
        if method == 'ucge':
            UCGEInitialize.initialize(qc, amplitude_array)
            return qc
        elif method == 'dcsp':
            qc = DcspInitialize(amplitude_array).definition
            return qc
        elif method == 'qiskit':
            gate = StatePreparation(amplitude_array)
            qc.append(gate, range(self.n_qubits))
            return qc

    def rx_layer(self, qc, param_pack):
        n = self.n_qubits
        for i in range(n):
            qc.rx(param_pack[i], i)
        return qc

    def cyclic_ansatz_layer(self, qc, param_pack):

        n = self.n_qubits

        for i in range(0, n - 1, 2):
            p = param_pack[i * 3: (i + 1) * 3]
            qc.append(CU3Gate(p[0], p[1], p[2]), [i, i + 1])


        for i in range(1, n - 1, 2):
            p = param_pack[i * 3: (i + 1) * 3]
            qc.append(CU3Gate(p[0], p[1], p[2]), [i, i + 1])

        if n > 1:
            p = param_pack[(n - 1) * 3: n * 3]
            qc.append(CU3Gate(p[0], p[1], p[2]), [n - 1, 0])

        return qc


    def ansatz(self,  params):

        qc = self._get_empty_circuit()
        n = self.n_qubits

        param_pack1 = params[:n]
        param_pack2 = params[n:4*n]
        param_pack3 = params[4*n:5*n]
        param_pack4 = params[5*n:8*n]
        param_pack5 = params[8*n:] # 9*n parameters

        self.rx_layer(qc, param_pack1)

        self.cyclic_ansatz_layer(qc, param_pack2)

        self.rx_layer(qc, param_pack3)

        self.cyclic_ansatz_layer(qc, param_pack4)

        self.rx_layer(qc, param_pack5)
        return qc


