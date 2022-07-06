import matplotlib.pyplot as plt
from qiskit.algorithms import VQE
from qiskit.utils import QuantumInstance
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.opflow import X, Z, I, Y, StateFn, PauliOp, OperatorBase, PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import ADAM, L_BFGS_B, CG, SPSA
from qiskit.circuit import parameterexpression, Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.circuit import Qubit
import numpy as np
from functools import reduce
from itertools import permutations, combinations
from math import factorial
import time
from scipy.special import lambertw
from typing import List, Union, Tuple, Optional


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
    # h = (Z ^ I ^ I) + (2 * I ^ Z ^ I) + (3 * I ^ I ^ Z)
    # h = (Z ^ I) + (10 * I ^ Z)
    # Error model parameters
    depol_error_rates = [0., 5e-2, 1e-1]
    np.random.seed(0)
    depol_error_rates_2q = np.random.rand(n_qubits, n_qubits) * 0.05
    depol_error_rates_2q += depol_error_rates_2q.T

    samples_per_permutation = 10

    # Ansatz parameters
    spsa = SPSA(maxiter=1000)
    ansatz = TwoLocal(rotation_blocks=['ry', 'rx', 'ry'],
                      entanglement_blocks='cz',
                      # entanglement=[],
                      reps=2,
                      num_qubits=h.num_qubits)

    # params = ParameterVector('theta', n_qubits * 2)
    #
    # qreg = QuantumRegister(n_qubits)
    # ansatz = QuantumCircuit(qreg)
    # for i in range(n_qubits):
    #     ansatz.ry(params[2 * i], qreg[i])
    #     ansatz.rx(params[2 * i + 1], qreg[i])

    results = np.zeros((factorial(n_qubits), samples_per_permutation))
    for i, perm in enumerate(permutations(range(n_qubits))):
        # Be careful if you actually assign according to the inverse.
        print('permutation {0:}'.format(i))
        noise_model = NoiseModel()
        for j, qubitno in enumerate(perm):
            noise_model.add_quantum_error(depolarizing_error(depol_error_rates[j], 1),
                                          ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'], [qubitno])
        for (j, k) in combinations(range(n_qubits), 2):
            noise_model.add_quantum_error(depolarizing_error(depol_error_rates_2q[j, k], 2),
                                          ['cz'], [perm[j], perm[k]])

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


def solve_then_check_all_perms(hamiltonian: PauliSumOp,
                               circ: QuantumCircuit,
                               noise_model: Optional[NoiseModel] = None):
    num_qubits = hamiltonian.num_qubits
    quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"),
                                       shots=1024,
                                       noise_model=noise_model)
    spsa = SPSA(maxiter=1000)
    vqe = VQE(circ,
              optimizer=spsa,
              quantum_instance=quantum_instance,
              initial_point=np.random.randn(circ.num_parameters) * 1e-4)
    sol = vqe.compute_minimum_eigenvalue(hamiltonian)
    perm_data = np.zeros(factorial(num_qubits))
    print('trivial permutation')
    print('E={0:3.3f}'.format(sol.eigenvalue))
    for i, perm in enumerate(permutations(range(num_qubits))):
        h_perm = hamiltonian.permute(list(perm))
        circ_perm = permute_circuit(circ, perm)
        vqe = VQE(circ_perm, quantum_instance=quantum_instance)
        en_eval = vqe.get_energy_evaluation(h_perm)
        perm_data[i] = en_eval(sol.optimal_point)
        print(perm)
        print(perm_data[i])




def permute_circuit(circ: QuantumCircuit,
                    perm: Union[List[int], Tuple[int]]) -> QuantumCircuit:
    """Return a QuantumCircuit in which every gate from circ is replaced
    by the same gate acting on qubits perm[i], perm[j], etc."""
    circ_new = QuantumCircuit(*circ.qregs)
    for gate in circ.data:
        instruction = gate[0]
        qubits = gate[1]
        params = gate[2]
        qubits_new = []
        for qubit in qubits:
            # print(qubit)
            # print(dir(qubit))
            # print(qubit.register)
            # print(qubit.index)
            qubits_new.append(Qubit(qubit.register, perm[qubit.index]))
        circ_new.data.append((instruction, qubits_new, params))
    return circ_new


def permute_hamiltonian(hamiltonian: PauliSumOp,
                        perm: Union[List[int], Tuple[int]]) -> PauliSumOp:
    reverse = list(range(hamiltonian.num_qubits - 1, -1, -1))
    return hamiltonian.permute(reverse).permute(perm).permute(reverse)


def plot_permutations_experiment(expt_time: str):
    data = np.loadtxt("permutation_vqe_" + expt_time + ".txt")
    n_qubits = int(round(inverse_factorial(data.shape[0]).real - 1))
    print(n_qubits)
    print(data.shape)
    for i in range(data.shape[1]):
        plt.scatter(np.arange(data.shape[0]), data[:, i], alpha=0.3, color="tab:blue")
    plt.boxplot(data.T, flierprops={"marker": ".",}, whis=(10, 90), positions=np.arange(data.shape[0]))
    perms = list(permutations(range(n_qubits)))
    print(perms)
    plt.xticks(np.arange(data.shape[0]), perms)
    plt.ylabel("VQE energy")
    plt.grid()
    plt.savefig("permutations_" + expt_time + ".png", format='png', bbox_inches='tight', dpi=400)
    plt.show()


if __name__ == '__main__':
    # all_permutations_experiment()
    # plot_permutations_experiment("1656335774")
    # print(inverse_factorial(factorial(5)))
    # h = (Z ^ I) + 5 * (I ^ Z)
    # ansatz = TwoLocal(rotation_blocks=['ry', 'rx', 'ry'],
    #                   entanglement_blocks='cz',
    #                   # entanglement=[],
    #                   reps=2,
    #                   num_qubits=h.num_qubits)
    #
    # depol_error_rates = [0., 1e-1]
    # np.random.seed(0)
    # # depol_error_rates_2q = np.random.rand(h.num_qubits, h.num_qubits) * 0.05
    # # depol_error_rates_2q += depol_error_rates_2q.T
    # noise_model = NoiseModel()
    # for j in range(h.num_qubits):
    #     noise_model.add_quantum_error(depolarizing_error(depol_error_rates[j], 1),
    #                                   ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'], [j])
    # # for (j, k) in combinations(range(h.num_qubits), 2):
    #     # noise_model.add_quantum_error(depolarizing_error(depol_error_rates_2q[j, k], 2),
    #     #                               ['cz'], [j, k])
    #
    # solve_then_check_all_perms(h, ansatz, noise_model)
    pass
