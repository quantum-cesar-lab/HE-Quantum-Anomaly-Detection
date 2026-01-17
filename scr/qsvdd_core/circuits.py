from qclib.state_preparation.ucge import UCGEInitialize


def feature_mapping(qc, amplitude_array):
    qc = UCGEInitialize.initialize(qc, amplitude_array)


