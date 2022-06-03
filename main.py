from qiskit.algorithms import VQE
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import X, Z, I, Y, StateFn
from qiskit.algorithms.optimizers import ADAM, L_BFGS_B, CG, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow.gradients import Gradient

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

import numpy as np

if __name__ == '__main__':
    h = -1 * Z ^ Z ^ I - 1 * I ^ Z ^ Z - 1 * X ^ X ^ X
    # h = Z ^ Z
    adam = ADAM(maxiter=100)
    bfgs = L_BFGS_B(maxfun=100)
    optimizer = SPSA(maxiter=100)
    ansatz = TwoLocal(rotation_blocks=['ry'], entanglement_blocks='cz', reps=3)

    grad = Gradient(grad_method='param_shift')

    noise_model = NoiseModel()
    error_rates = [1e-3, 1e-2, 2e-2]
    for i, rate in enumerate(error_rates):
        noise_model.add_quantum_error(depolarizing_error(rate, 1),
                                      ['u1', 'u2', 'u3'], [i])

    quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"),
                                       shots=1024,
                                       noise_model=noise_model)

    vqe = VQE(ansatz,
              optimizer=bfgs,
              quantum_instance=quantum_instance,
              gradient=grad)

    result = vqe.compute_minimum_eigenvalue(operator=h)
    print(result.eigenvalue)
    print(result.optimizer_time)
