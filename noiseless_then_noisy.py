from main import *
from qiskit.providers.aer import AerSimulator, AerProvider


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


def clean_solution(h: PauliSumOp, ansatz: QuantumCircuit):
    clean_instance = QuantumInstance(backend=AerProvider().get_backend('aer_simulator_statevector'))
    vqe = VQE(ansatz,
              optimizer=L_BFGS_B(),
              quantum_instance=clean_instance,
              initial_point=np.random.randn(ansatz.num_parameters) * 1e-4,
              include_custom=True)
    result_clean = vqe.compute_minimum_eigenvalue(h)
    return result_clean


if __name__ == "__main__":
    h = (Z ^ I) + 5 * (I ^ Z)
    ansatz = TwoLocal(rotation_blocks=['ry', 'rx', 'ry'],
                      entanglement_blocks='rxx',
                      # entanglement=[],
                      reps=1,
                      num_qubits=h.num_qubits)

    depol_error_rates = [0., 1e-1]
    np.random.seed(0)
    noise_model = NoiseModel()
    for j in range(h.num_qubits):
        noise_model.add_quantum_error(depolarizing_error(depol_error_rates[j], 1),
                                      ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'], [j])

    clean_then_noisy_clean(h, ansatz, noise_model, all_perms=True)

