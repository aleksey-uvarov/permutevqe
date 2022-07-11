import random
import matplotlib.pyplot as plt
import numpy as np
from qiskit.providers.aer import AerSimulator, AerProvider
from qiskit import execute
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.opflow import OperatorBase
from qiskit.providers.aer.backends import QasmSimulator
from scipy.optimize import minimize, OptimizeResult
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQEResult, MinimumEigensolverResult
from qiskit.quantum_info.operators.symplectic import Pauli

from main import *


PAULI_LABELS = ('I', 'X', 'Y', 'Z')


def clean_then_noisy_clean(h: PauliSumOp,
                           ansatz: QuantumCircuit,
                           noise_model: NoiseModel,
                           all_perms=True):
    result_clean = clean_solution(h, ansatz)
    noisy_backend = AerProvider().get_backend('aer_simulator_density_matrix')
    noisy_instance = QuantumInstance(backend=noisy_backend,
                                     noise_model=noise_model)
    if all_perms:
        results_energies = np.zeros(factorial(h.num_qubits))
        for i, perm in enumerate(permutations(range(h.num_qubits))):
            h_perm = h.permute(list(perm))
            ansatz_new = permute_circuit(ansatz, perm)
            vqe_perm = VQE(ansatz_new,
                           optimizer=L_BFGS_B(),
                           quantum_instance=noisy_instance,
                           initial_point=result_clean.optimal_point,
                           include_custom=True)
            result_noisy = vqe_perm.compute_minimum_eigenvalue(h_perm)
            print(result_noisy.eigenvalue)
            print(perm)


def clean_solution(h: OperatorBase, ansatz: QuantumCircuit) -> VQEResult:
    clean_instance = QuantumInstance(backend=AerProvider().get_backend('aer_simulator_statevector'))
    vqe = VQE(ansatz,
              optimizer=L_BFGS_B(),
              quantum_instance=clean_instance,
              initial_point=np.random.randn(ansatz.num_parameters) * 1e-4,
              include_custom=True)
    result_clean = vqe.compute_minimum_eigenvalue(h)
    return result_clean


def telescope_hamiltonian(ansatz: QuantumCircuit, seed: Optional[int] = None) -> MatrixOp:
    n_qubits = ansatz.num_qubits
    pauli_list = [Z] + [I] * (n_qubits - 1)
    h_0 = -0.5 * reduce(PauliOp.tensor, pauli_list)
    for i in range(1, n_qubits):
        h_0 += -0.5 * reduce(PauliOp.tensor, pauli_list[i:] + pauli_list[:i])
    h_0 += reduce(PauliOp.tensor, [I] * n_qubits) * n_qubits * 0.5
    backend = Aer.get_backend("unitary_simulator")
    rng = np.random.default_rng(seed)
    circ = ansatz.bind_parameters(rng.random(ansatz.num_parameters) * 2 * np.pi)
    result = execute(experiments=circ, backend=backend).result()
    U = result.get_unitary()
    h = MatrixOp(U @ h_0.to_matrix() @ U.T.conj())
    return h


def telescope_optimization_experiment(n_qubits: int,
                                      depth: int,
                                      noise_model: Optional[NoiseModel] = None,
                                      telescope_seed: Optional[int] = None):
    ansatz = TwoLocal(rotation_blocks=['ry', 'rx', 'ry'],
                      entanglement_blocks='rxx',
                      reps=depth,
                      num_qubits=n_qubits)
    h = telescope_hamiltonian(ansatz, seed=telescope_seed)
    sol_clean = clean_solution(h, ansatz)
    print('clean', sol_clean.eigenvalue)
    print(sol_clean.optimal_point)

    perms_data = np.zeros(factorial(n_qubits) + 1)
    for i, perm in enumerate(permutations(range(n_qubits))):
        h_perm = h.permute(list(perm))
        ansatz_perm = permute_circuit(ansatz, perm)
        foo = get_energy_estimator(ansatz_perm, h_perm, noise_model)
        print("Initial guess quality ", foo(sol_clean.optimal_point))
        sol = minimize(foo, sol_clean.optimal_point, method='L-BFGS-B')
        print(perm, sol.fun)
        print(sol)
        perms_data[i] = sol.fun
    perms_data[-1] = sol_clean.eigenvalue

    np.savetxt("permutation_vqe_" + str(int(time.time())) + ".txt", perms_data)


def get_energy_estimator(ansatz: QuantumCircuit, h: np.array, noise_model: NoiseModel):
    def f(x):
        noisy_backend = QasmSimulator(method='density_matrix',
                                      noise_model=noise_model)
        ansatz_2 = ansatz.copy()
        ansatz_2.save_density_matrix()
        ansatz_2 = ansatz_2.bind_parameters(x)
        result = execute(ansatz_2, noisy_backend).result()
        dm = result.data()['density_matrix']
        return np.trace(dm.data @ h.to_matrix()).real
    return f


def plot_telescope_experiment(expt_time: str):
    data = np.loadtxt("permutation_vqe_telescope_" + expt_time + ".txt")
    n_qubits = int(round(inverse_factorial(data.shape[0] - 1).real - 1))
    print(n_qubits)
    print(data.shape)
    plt.scatter(np.arange(data.shape[0] - 1), data[:-1])
    perms = list(permutations(range(n_qubits)))
    print(perms)
    plt.axhline(y=data[-1])
    plt.xticks(np.arange(data.shape[0] - 1), perms)
    plt.ylabel("VQE energy")
    plt.grid()
    plt.savefig("permutations_vqe_telescope_" + expt_time + ".png", format='png', bbox_inches='tight', dpi=400)
    plt.show()


def pauli_op_optimization_experiment(n_qubits: int,
                                     depth: int,
                                     qty_strings: int,
                                     noise_model: Optional[NoiseModel] = None,
                                     seed: Optional[int] = None):
    ansatz = TwoLocal(rotation_blocks=['ry', 'rx', 'ry'],
                      entanglement_blocks='rxx',
                      reps=depth,
                      num_qubits=n_qubits)
    h = random_pauli_op(n_qubits, qty_strings, seed)
    sol_clean = clean_solution(h, ansatz)
    print('clean', sol_clean.eigenvalue)
    print(sol_clean.optimal_point)

    perms_data = np.zeros(factorial(n_qubits) + 1)
    initial_guess_qualities = np.zeros_like(perms_data)
    for i, perm in enumerate(permutations(range(n_qubits))):
        h_perm = h.permute(list(perm))
        ansatz_perm = permute_circuit(ansatz, perm)
        foo = get_energy_estimator(ansatz_perm, h_perm, noise_model)
        print("Initial guess quality ", foo(sol_clean.optimal_point))
        initial_guess_qualities[i] = foo(sol_clean.optimal_point)
        sol = minimize(foo, sol_clean.optimal_point, method='L-BFGS-B')
        print(perm, sol.fun)
        print(sol)
        perms_data[i] = sol.fun
    perms_data[-1] = sol_clean.eigenvalue

    timestamp = str(int(time.time()))
    np.savetxt("permutation_pauli_op_" + timestamp + ".txt", perms_data)
    np.savetxt("initguess_pauli_op_" + timestamp + ".txt", initial_guess_qualities)


def random_pauli_op(num_qubits: int, qty_strings: int, seed: Optional[int] = None) -> PauliSumOp:
    """Prepares a Hamiltonian containing qty_strings random Pauli strings on num_qubits qubits,
    weighted with coefficients drawn from the standard normal distribution."""
    random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    pool_size = 0
    labels_pool = []
    if qty_strings > 4**num_qubits:
        raise ValueError('Too many Pauli strings')

    while pool_size < qty_strings:
        pauli_labels = ''.join([random.choice(PAULI_LABELS) for _ in range(num_qubits)])
        if pauli_labels not in labels_pool:
            labels_pool.append(pauli_labels)
            pool_size += 1

    h = rng.normal() * PauliOp(Pauli(labels_pool[0]))
    for i in range(1, qty_strings):
        h += rng.normal() * PauliOp(Pauli(labels_pool[i]))
    print(h)
    return h


if __name__ == "__main__":
    n_qubits = 3
    depth = 2
    qty_strings = 10
    local_errors = [0., 1e-3, 1e-2]
    seed = 0
    noise_model = NoiseModel()
    for j in range(n_qubits):
        noise_model.add_quantum_error(depolarizing_error(local_errors[j], 1), ['rx', 'ry'], [j])
    pauli_op_optimization_experiment(n_qubits, depth, qty_strings)

    # telescope_optimization_experiment(n_qubits, depth, noise_model, seed)

    # plot_telescope_experiment('1657197856')

    # random_pauli_op(3, 10, seed=0)
