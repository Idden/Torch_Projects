import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class HadamardTransform:
    def __call__(self, dataset):

        # flatten image and pad to 2^n length
        img_flat = dataset.flatten().numpy()
        n_qubits = int(np.ceil(np.log2(len(img_flat))))
        padded_len = 2 ** n_qubits
        padded_image = np.zeros(padded_len)
        padded_image[:len(img_flat)] = img_flat

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        # get the statevector
        backend = AerSimulator('statevector_simulator')
        result = transpile(qc, backend)
        statevector = result.get_statevector()

        print(statevector)

        # combine with original amplitudes
        transformed = np.abs(statevector[:len(img_flat)])
        transformed = transformed.reshape(dataset.shape)
        return transformed