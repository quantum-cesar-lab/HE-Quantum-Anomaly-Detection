from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp


class QuantumEngine:
    def __init__(self):
        self.estimator = Estimator()

    def get_expectation_value(self, circuit, params):
        """
        Calcula a correlação entre qubits
        """
        observables = [
            SparsePauliOp("IIXIX"),  # Correlação X(0)X(2)
            SparsePauliOp("IIYIY"),  # Correlação Y(0)Y(2)
            SparsePauliOp("IIZIZ"),  # Correlação Z(0)Z(2)
        ]

        job = self.estimator.run([circuit] * 3, observables)
        return job.result().values
