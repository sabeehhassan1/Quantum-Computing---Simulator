import numpy as np
from quantum_circuit import *


class GroverCircuit(QuantumCircuit):
    def apply_conditional_phase_shift(self):
        for state in range(1, self.num_states):
            self.state_vector[state] = -1 * self.state_vector[state]
        return self

    def apply_oracle(self):
        correct_state = 128
        self.state_vector[correct_state] = -1 * self.state_vector[correct_state]
        return self

circuit = GroverCircuit(8)
circuit.hadamard()

num_cycles = np.pi / 4 * np.sqrt(2**circuit.num_qubits)
print('Number of cycles:', num_cycles)

for _ in range(int(np.round(num_cycles))):
    circuit.apply_oracle().hadamard().apply_conditional_phase_shift().hadamard()

print('Measured state:', circuit.measure())
