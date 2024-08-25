from quantum_circuit import *
from quantum_memory import *

circuit = QuantumCircuit(3)

circuit.apply_gate(1, 1./np.sqrt(2), 1./np.sqrt(2))
circuit.apply_gate(2, 1./np.sqrt(3), np.sqrt(2./3.))

circuit.hadamard(qubits=[1, 0, 0])
circuit.controlled_swap(0, 1, 2)
circuit.hadamard(qubits=[1, 0, 0])

def avg(values):
    return sum(values) / float(len(values))

eps = 1e-4
average = avg([circuit.measure() < 4 for _ in range(int(eps**(-1)))])

print("Measure avg", np.sqrt(average * 2 - 1))
print("Probability", circuit.measure_probability([0, None, None]))
print("Correct    ", 1./np.sqrt(2)*1./np.sqrt(3) + 1./np.sqrt(2)*np.sqrt(2./3.))
