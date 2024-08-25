import numpy as np
import numpy.linalg as lg
import numpy.random as random
import matplotlib.pyplot as plt

from quantum_circuit import *


class MinimumCircuit(QuantumCircuit):
    def __init__(self, num_qubits, func):
        super(MinimumCircuit, self).__init__(num_qubits)

        self.func = func

        random_state = random.randint(self.num_states)
        self.threshold_value = func(random_state)
        self.threshold_state = random_state

    def apply_conditional_phase_shift(self):
        for state in range(1, self.num_states):
            self.state_vector[state] = -1 * self.state_vector[state]
        return self

    def apply_oracle(self):
        for state in range(self.num_states):
            if self.func(state) < self.threshold_value:
                self.state_vector[state] = -1 * self.state_vector[state]
        return self

    def calculate_cycles(self):
        return np.pi / 4 * np.sqrt(2 ** self.num_qubits)

    def perform_search(self):
        self.hadamard()
        for _ in range(int(np.round(self.calculate_cycles()))):
            self.apply_oracle().hadamard().apply_conditional_phase_shift().hadamard()

        return self.measure()

if __name__ == '__main__':
    circuit = MinimumCircuit(8, lambda x: (x / 128. - 1) ** 2)

    estimated_cycles = 22.5 * np.sqrt(circuit.num_states) + 1.4 * np.log10(circuit.num_states) ** 2

    print('Estimated cycles:', int(np.round(estimated_cycles)))
    iterations = 0
    total_runtime = 0

    while total_runtime < estimated_cycles:
        circuit.reset()
        measured_state = circuit.perform_search()
        measured_value = circuit.func(measured_state)

        if measured_value < circuit.threshold_value:
            circuit.threshold_value = measured_value
            circuit.threshold_state = measured_state

        iterations += 1
        total_runtime += circuit.calculate_cycles()

        print('Best state:', circuit.threshold_state, 'Best value:', circuit.threshold_value, 'Measured value:', measured_value)

    print('Final best state:', circuit.threshold_state)
    print('Total iterations:', iterations)
    print('Total runtime:', total_runtime)
    print('Measured state:', circuit.threshold_state)
    print('Measured value:', circuit.func(circuit.threshold_state))
