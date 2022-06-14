import matplotlib.pyplot as plt
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
from scipy.special import lambertw


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
    h = circular_chiral_walk(n_qubits, np.pi / 2)

    # Error model parameters
    depol_error_rates = [1e-3, 1e-2, 2e-2]

    samples_per_permutation = 10

    # Ansatz parameters
    spsa = SPSA(maxiter=1000)
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
            print(result.optimizer_time)
    print(results)
    np.savetxt("permutation_vqe_" + str(int(time.time())) + ".txt", results)


def inverse_factorial(x: int) -> complex:
    # mathoverflow.net/questions/12828
    c = 0.036534
    L = np.log((x + c) / (2 * np.pi)**0.5)
    return L / lambertw(L / np.e) + 0.5


def plot_permutations_experiment(expt_time: str):
    data = np.loadtxt("permutation_vqe_" + expt_time + ".txt")
    n_qubits = int(inverse_factorial(data.shape[0]).real)
    print(data.shape)
    for i in range(data.shape[1]):
        plt.scatter(np.arange(data.shape[0]), data[:, i], alpha=0.3, color="tab:blue")
    plt.boxplot(data.T, flierprops={"marker": ".",}, whis=(10, 90), positions=np.arange(data.shape[0]))
    perms = list(permutations(range(n_qubits)))
    print(perms)
    plt.xticks(np.arange(data.shape[0]), perms)
    plt.ylabel("VQE energy")
    plt.savefig("permutations_" + expt_time + ".png", format='png', bbox_inches='tight', dpi=400)
    plt.show()


if __name__ == '__main__':
    # all_permutations_experiment()
    plot_permutations_experiment("1655203364")
    # print(inverse_factorial(factorial(5)))
