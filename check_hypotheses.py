import matplotlib.pyplot as plt
import numpy as np
from noiseless_then_noisy import *


def local_score(H, p, depth):
    fids_vector = (1 - p)**(3 * depth)
    score = 0
    for s in H:
        coeff = s.primitive.coeffs
        pauli_string = s.primitive.paulis[0][::-1]
        multiplier = reduce(float.__mul__, [fids_vector[i] for i, pauli in enumerate(pauli_string)
                                            if str(pauli) != 'I'])
        score += coeff * multiplier
    return score.real
    # doesn't look like it correlates with the energy


if __name__ == "__main__":
    timestamp = '1657709565'
    seed = 0
    num_qubits = 3
    depth = 2
    local_errors = np.array([0., 1e-3, 1e-2])
    qty_strings = 10

    data = np.loadtxt("permutation_pauli_op_" + timestamp + ".txt")
    H = random_pauli_op(num_qubits, qty_strings, seed)

    scores = np.zeros(factorial(num_qubits))
    for i, perm in enumerate(permutations(range(num_qubits))):
        print(perm)
        permuted_errors = np.array([local_errors[j] for j in perm])
        print(permuted_errors)
        scores[i] = local_score(H, permuted_errors, depth)

    plt.scatter(scores, data[:-1])
    plt.show()
