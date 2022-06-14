from qiskit.algorithms import VQE
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import X, Z, I, Y, StateFn, PauliOp, OperatorBase
from qiskit.algorithms.optimizers import ADAM, L_BFGS_B, CG, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow.gradients import Gradient
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error
import numpy as np
from functools import reduce
from itertools import permutations
from math import factorial
import time


def circular_chiral_walk(n_sites: int, alpha: float):
    hamiltonian = 0.
    xx_list = [X, X] + [I] * (n_sites - 2)
    xy_list = [X, Y] + [I] * (n_sites - 2)
    yx_list = [Y, X] + [I] * (n_sites - 2)
    yy_list = [Y, Y] + [I] * (n_sites - 2)
    for i in range(n_sites):
        hamiltonian += 0.5 * np.cos(alpha) * reduce(PauliOp.tensor, xx_list[i:] + xx_list[:i])
        hamiltonian += -0.5 * np.sin(alpha) * reduce(PauliOp.tensor, xy_list[i:] + xy_list[:i])
        hamiltonian += 0.5 * np.sin(alpha) * reduce(PauliOp.tensor, yx_list[i:] + yx_list[:i])
        hamiltonian += 0.5 * np.cos(alpha) * reduce(PauliOp.tensor, yy_list[i:] + yy_list[:i])
    return hamiltonian


def all_permutations_experiment() -> None:
    # Problem Hamiltonian
    n_qubits = 3
    h = circular_chiral_walk(n_qubits, 0.1)

    # Error model parameters
    depol_error_rates = [1e-3, 1e-2, 2e-2]

    samples_per_permutation = 1

    # Ansatz parameters
    spsa = SPSA(maxiter=100)
    ansatz = TwoLocal(rotation_blocks=['ry'],
                      entanglement_blocks='cz',
                      reps=2,
                      num_qubits=h.num_qubits)

    results = np.zeros((factorial(n_qubits), samples_per_permutation))
    for i, perm in enumerate(permutations(range(n_qubits))):
        print('permutation {0:}'.format(i))
        noise_model = NoiseModel()
        for j, qubitno in enumerate(perm):
            noise_model.add_quantum_error(depolarizing_error(depol_error_rates[j], 1),
                                          ['u1', 'u2', 'u3'], [qubitno])

        quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"),
                                           shots=1024,
                                           noise_model=noise_model)

        vqe = VQE(ansatz,
                  optimizer=spsa,
                  quantum_instance=quantum_instance,
                  initial_point=np.random.randn(ansatz.num_parameters) * 1e-4)

        for j in range(samples_per_permutation):
            print('sample {0:}'.format(j))
            result = vqe.compute_minimum_eigenvalue(operator=h)
            results[i, j] = result.eigenvalue
            print(result.eigenvalue)
    print(results)
    np.savetxt("permutation_vqe" + str(int(time.time())) + ".txt", results)


if __name__ == '__main__':
    all_permutations_experiment()
    # h = -1 * (Z ^ Z ^ I) - 1 * (I ^ Z ^ Z) - 1 * (X ^ X ^ X) # careful with the brackets!
    # h = circular_chiral_walk(3, 0.1)
    # # print(h)
    # adam = ADAM(maxiter=100)  # uses jacobian but is slow for some reason
    # bfgs = L_BFGS_B(maxfun=100)  # passes jacobian to scipy
    # spsa = SPSA(maxiter=100)  # doesn't use partial derivative
    # ansatz = TwoLocal(rotation_blocks=['ry'], entanglement_blocks='cz', reps=3, num_qubits=h.num_qubits)
    #
    # grad = Gradient(grad_method='param_shift')
    #
    # noise_model = NoiseModel()
    # error_rates = [1e-3, 1e-2, 2e-2]
    # for i, rate in enumerate(error_rates):
    #     noise_model.add_quantum_error(depolarizing_error(rate, 1),
    #                                   ['u1', 'u2', 'u3'], [i])
    #
    # print(noise_model)
    # all_permutations_experiment(h, noise_model)

    # quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"),
    #                                    shots=1024,
    #                                    noise_model=noise_model)
    #
    # vqe = VQE(ansatz,
    #           optimizer=spsa,
    #           quantum_instance=quantum_instance,
    #           gradient=grad,
    #           initial_point=np.random.randn(ansatz.num_parameters) * 1e-4)
    #
    # result = vqe.compute_minimum_eigenvalue(operator=h)
    # print("spsa")
    # print(result.eigenvalue)
    # print(result.optimizer_time)
