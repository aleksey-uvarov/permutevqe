"""Here we analyze the permutation variability of the
VQE solution. We already established that the sum of error rates
is a nice-ish proxy for fidelity. Now we want to find out whether
this quantity is also a nice thing to look out for when running VQE.
To do that, we take a Hamiltonian, run VQE, then for every permutation,
we estimate the error sum, the fidelity, and the energy of the solution.

So far the answer is no"""
import matplotlib.pyplot as plt

from main import *
from qiskit_aer.backends import QasmSimulator, AerSimulator
from tqdm import tqdm
import random
from qiskit.quantum_info.operators.symplectic import Pauli
from typing import Callable
from qiskit_aer.noise import depolarizing_error
from qiskit_aer import noise
from qiskit.opflow.gradients import Gradient, NaturalGradient
from qiskit.algorithms.optimizers import ADAM, L_BFGS_B, CG, SPSA, GradientDescent
from enum import Enum, auto
from qiskit import execute


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Hamiltonian(Enum):
    ISING = auto()
    PERTURBED_ISING = auto()

    
class NoiseType(Enum):
    DEPOL = auto()
    BITFLIP = auto()
    PHASEFLIP = auto()
    MS = auto()
    DEPOL_2 = auto()
    
    
def get_energy_estimator(ansatz: QuantumCircuit, 
                         h: PauliSumOp, 
                         noise_model: NoiseModel):
    """Energy estimator function that treats the noisy state as a density matrix and
    performs all calculations exactly. Use when 
    you don't want the optimization process
    to be realistic, but are interested in the optimization landscape itself."""
    def f(x):
        noisy_backend = AerSimulator(method='density_matrix',
                                      noise_model=noise_model)
        ansatz_2 = ansatz.copy()
        ansatz_2.save_density_matrix()
        ansatz_2 = ansatz_2.bind_parameters(x)
        result = execute(ansatz_2, noisy_backend).result()
        dm = result.data()['density_matrix']
        return np.trace(dm.data @ h.to_matrix()).real
    return f


def ising_model(n_spins, J=1, hx=1):
    ham = {}
    line = 'Z' + 'Z' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = J
    line = 'X' + 'I' * (n_spins - 1)
    if hx != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = hx
    return dictionary_to_pauliop(ham)


def perturbed_ising_model(n_spins: int, 
                          j_avg: float = 1,
                          j_var: float = 0.5,
                          h_avg: float = 1,
                          h_var: float = 0.5,
                          rng_seed: int = None):
    rng_local = np.random.default_rng(rng_seed)
    js = rng_local.normal(j_avg, j_var, size=n_spins)
    hs = rng_local.normal(h_avg, h_var, size=n_spins)
    ham = {}
    line = 'Z' + 'Z' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = js[i]
    line = 'X' + 'I' * (n_spins - 1)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = hs[i]
    return dictionary_to_pauliop(ham)
    
    

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
    # print(h)
    return h


def error_sum(circ: QuantumCircuit,
              ps: np.array, qs: np.array) -> float:
    """We have this method for 'TwoLocal' circuits, which
    requires some stupid specialization"""
    error_rate_total = 0
    for true_gate in circ.data:
        qubits = true_gate.qubits
        if len(qubits) == 2:
            index_1, index_2 = qubits[0].index, qubits[1].index
            error_rate_total += qs[index_1, index_2]
        elif len(qubits) == 1:
            error_rate_total += ps[qubits[0].index]
    return error_rate_total


def unpack_twolocal(circ: QuantumCircuit) -> QuantumCircuit:
    circ_unpacked = QuantumCircuit(*circ.qregs)

    for gate in circ:
        defn = gate.operation.definition
        if 'data' not in dir(defn):
            continue
        for true_gate in defn.data:
            circ_unpacked.data.append(true_gate)

    return circ_unpacked


def measure_h_with_errorbar(h: PauliSumOp,
                             circ: QuantumCircuit,
                             noise_model: NoiseModel,
                             shots: int) -> (float, float):
    backend = Aer.get_backend("qasm_simulator")
    # Measure in density matrices?
    # But then vqe was in sort of realistic conditions?

    return 0., 0.


def classical_perm_criterion(circ: QuantumCircuit,
                             h: PauliSumOp,
                             criterion: Callable):
    ...


if __name__ == '__main__':

    n_qubits = 8
    depth = 3
    p_magnitude = 0.
    q_magnitude = 1e-2
    shots = 1024
    maxiter = 100

    backend_clean = Aer.get_backend('statevector_simulator')
    timestamp = int(time.time())

    h = ising_model(n_qubits, 1, 1)
    h_square = h.compose(h)

    ps = abs(np.random.randn(n_qubits) * p_magnitude)
    qs = abs(np.random.randn(n_qubits, n_qubits) * q_magnitude)
    qs = np.triu(qs, 1) + np.triu(qs, 1).T

    # spsa = SPSA(maxiter=1000)
    noise_model = NoiseModel()
    for j in range(n_qubits):
        # localerror = noise.phase_damping_error(ps[j])
        localerror = noise.depolarizing_error(ps[j], 1)
        # localerror = noise.pauli_error([('X', ps[j]), ('I', 1 - ps[j])])
        noise_model.add_quantum_error(localerror,
                                      ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'], [j])
    for i in range(n_qubits):
        for j in range(n_qubits):
            A0 = np.array([[1, 0], [0, (1 - qs[i, j]) ** 0.5]])
            A2 = np.array([[0, 0], [0, (qs[i, j]) ** 0.5]])

            # localerror = noise.kraus_error(
            #     [np.kron(A0, A0),
            #      np.kron(A0, A2),
            #      np.kron(A2, A0),
            #      np.kron(A2, A2)]
            # )

            # localerror = noise.pauli_error(
            #     [('II', (1 - qs[i, j])**2),
            #      ('IX', qs[i, j] * (1 - qs[i, j])),
            #      ('XI', qs[i, j] * (1 - qs[i, j])),
            #      ('XX', qs[i, j]**2)
            #      ])
            localerror = depolarizing_error(qs[i, j], 2)
            noise_model.add_quantum_error(localerror,
                                          ['rxx'], [i, j])
    backend = QasmSimulator(method='density_matrix',
                            noise_model=noise_model)

    circ = TwoLocal(n_qubits, ['ry'], 'rxx',
                    entanglement='linear',
                    reps=depth)
    circ = unpack_twolocal(circ)
    # circ.save_density_matrix()

    error_sums = np.zeros(factorial(n_qubits))
    energies = np.zeros(factorial(n_qubits))
    stdevs = np.zeros(factorial(n_qubits))
    out_data = np.zeros((factorial(n_qubits), n_qubits + 3))

    quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"),
                                       shots=shots,
                                       noise_model=noise_model)

    for i, perm in tqdm(enumerate(permutations(range(n_qubits)))):
        out_data[i, :-3] = perm
        h_perm = h.permute(list(perm))
        circ_perm = permute_circuit(circ, perm)

        # blindly copying from qiskit docs on gradient-based vqe
        # https://qiskit.org/documentation/stable/0.38/tutorials/operators/02_gradients_framework.html?highlight=vqe

        op_perm = ~StateFn(h_perm) @ StateFn(circ_perm)
        grad = Gradient(grad_method='param_shift')
        optimizer = CG(maxiter=maxiter)

        vqe = VQE(circ_perm,
                  optimizer=optimizer,
                  gradient=grad,
                  quantum_instance=quantum_instance,
                  initial_point=np.random.randn(circ.num_parameters) * 1e-4)
        sol = vqe.compute_minimum_eigenvalue(h_perm)

        en_eval = vqe.get_energy_evaluation(h_perm)
        sq_eval = vqe.get_energy_evaluation(h_perm.compose(h_perm))
        energies[i] = en_eval(sol.optimal_point)
        sq = sq_eval(sol.optimal_point)
        stdevs[i] = (sq - energies[i]**2)**0.5 / shots**0.5
        error_sums[i] = error_sum(circ_perm, ps, qs)

    out_data[:, -3] = energies
    out_data[:, -2] = stdevs
    out_data[:, -1] = error_sums

    print(out_data)
    np.savetxt("permutations_en_sum_{0:}.txt".format(timestamp), out_data)

    expt_data = {"n_qubits": n_qubits,
                 "depth": depth,
                 "p_magnitude": p_magnitude,
                 "q_magnitude": q_magnitude,
                 "shots": shots,
                 "error_type": "bit flip"}
                 # "hamiltonian_seed": seed,
                 # "hamiltonian_cardinality": card}
    with open("data/ising_data_{0:}.json".format(timestamp), "w") as fp:
        json.dump(expt_data, fp)
    np.savetxt("data/ps_{0:}.txt".format(timestamp), ps)
    np.savetxt("data/qs_{0:}.txt".format(timestamp), qs)

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(9, 6))
    plt.errorbar(error_sums, energies, stdevs, None, 'o', capsize=5)
    plt.xlabel('Error sum')
    plt.ylabel('E')
    plt.savefig('data/ising_perm_{0:}.png'.format(timestamp),
                format='png',
                bbox_inches='tight', dpi=400)
    plt.show()
