from qiskit import Aer, execute
from copy import copy, deepcopy

from vqe_vs_sum import *

sv_backend = Aer.get_backend("statevector_simulator")

def e_gs(circ, h, params, noise_type):
    """Return E_g for every gate g"""
    circ_bnd = circ.bind_parameters(params)
    names = [gate_name(gate) for gate in circ_bnd.data]
    indices = [i for i, x in enumerate(names) if x == "rxx" or x == "rzz"]
    qty_circs = 16 * len(indices)
    
    new_circs = get_intermediate_circuits(circ_bnd, 
                                          noise_type, 
                                          indices)
    
    if noise_type != NoiseType.MS:
    
        state_original = get_state(circ_bnd)
        states_with_errors = [get_state(c) for c in new_circs]
        Es_k = [state.expectation_value(h).real 
                for state in states_with_errors]
        batchmeans = [np.mean(Es_k[16 * i : 16 * (i + 1)]) 
                      for i in range(len(indices))]

        return batchmeans
    
    else:
        
        state_original = get_state(circ_bnd)
        states_with_errors = [get_state(c) for c in new_circs]
        Es_k = [state.expectation_value(h).real 
                for state in states_with_errors]
        # batchmeans = [np.mean(Es_k[16 * i : 16 * (i + 1)]) 
        #               for i in range(len(indices))]

        return Es_k
    
    
def get_state(circ):
    """Takes a circuit and returns a pure statevector"""
    job = execute(circ, sv_backend)
    result = job.result()
    state = result.get_statevector(circ)
    return state

    
def get_intermediate_circuits(circ_in, noisetype, indices):
    """Returns a list of circuits that appear when noise affects the gates"""
    new_circs = []

    if noisetype != NoiseType.MS:

        for i, index in enumerate(indices):
            for j in range(16):
                new_circ = deepcopy(circ_in)
                if noisetype == NoiseType.BITFLIP:
                    gates = [new_circ.id, new_circ.id, new_circ.x, new_circ.x]
                elif noisetype == NoiseType.PHASEFLIP:
                    gates = [new_circ.id, new_circ.id, new_circ.z, new_circ.z]
                elif noisetype == NoiseType.DEPOL:
                    gates = [new_circ.id, new_circ.x, new_circ.y, new_circ.z]
                rxx_qubits = [qbt.index for qbt in new_circ.data[index].qubits]
                gates[j % 4](rxx_qubits[0])
                gates[j // 4](rxx_qubits[1])
                new_circ.data.insert(index + 1, new_circ.data[-1])
                new_circ.data.insert(index + 1, new_circ.data[-2])
                new_circ.data.pop(-1)
                new_circ.data.pop(-1)
                new_circs.append(new_circ)

    else:
        for i, index in enumerate(indices):
            new_circ = deepcopy(circ_in)
            rxx_qubits = [qbt.index for qbt in new_circ.data[index].qubits]
            new_circ.x(rxx_qubits[0])
            new_circ.x(rxx_qubits[1])
            new_circ.data.insert(index + 1, new_circ.data[-1])
            new_circ.data.insert(index + 1, new_circ.data[-2])
            new_circ.data.pop(-1)
            new_circ.data.pop(-1)
            new_circs.append(new_circ)
    
    return new_circs


def gate_name(gate):
    return gate.operation.name