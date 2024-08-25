import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


class QuantumCircuit(object):
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0

    def reset(self):
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0

    # REGISTER MANIPULATION

    def is_set(self, state, qubit):
        return state & 1 << (self.num_qubits - 1 - qubit) != 0

    def flip_bit(self, state, qubit):
        return state ^ 1 << (self.num_qubits - 1 - qubit)

    def apply_gate(self, qubit, alpha, beta):  # alpha|0> + beta|1>
        temp_state_vector = np.zeros(self.num_states, dtype=complex)
        for state in range(self.num_states):
            current_amplitude = self.state_vector[state] + self.state_vector[self.flip_bit(state, qubit)]
            if self.is_set(state, qubit):
                temp_state_vector[state] = current_amplitude * beta
            else:
                temp_state_vector[state] = current_amplitude * alpha
        self.state_vector = temp_state_vector

    # MEASUREMENT OPERATIONS

    def measure(self):
        probabilities = np.absolute(self.state_vector)**2
        return random.choice(len(probabilities), p=probabilities.flatten())

    def measure_probability(self, target_qubits):
        assert len(target_qubits) == self.num_qubits
        prob = 0.0
        for state in range(self.num_states):
            selected = True
            for i in range(self.num_qubits):
                if target_qubits[i] is not None:
                    selected &= (self.is_set(state, i) == target_qubits[i])
            if selected:
                prob += np.absolute(self.state_vector[i])**2
            print(state, selected, prob)
        return prob

    # QUANTUM GATES

    def hadamard(self, target_qubits=None):
        if target_qubits is None:
            target_qubits = [1] * self.num_qubits
        H = 1. / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
        m = np.array([1])
        for indicator in reversed(target_qubits):
            m = np.kron(H, m) if indicator else np.kron(np.eye(2), m)
        self.state_vector = m.dot(self.state_vector)
        return self

    def hadamard_alternative(self):
        hadamard_matrix = np.zeros((self.num_states, self.num_states))
        for target in range(self.num_states):
            for state in range(self.num_states):
                hadamard_matrix[target, state] = (2.**(-self.num_qubits / 2.))*(-1)**bin(state & target).count("1")
        self.state_vector = hadamard_matrix.dot(self.state_vector)
        return self

    def controlled_swap(self, control_qubit, target_qubit_a, target_qubit_b):
        cswap_matrix = np.zeros((self.num_states, self.num_states))
        for state in range(self.num_states):
            if self.is_set(state, control_qubit):
                if self.is_set(state, target_qubit_a) != self.is_set(state, target_qubit_b):
                    flipped_state = self.flip_bit(self.flip_bit(state, target_qubit_b), target_qubit_a)
                    cswap_matrix[state, flipped_state] = 1.0
                else:
                    cswap_matrix[state, state] = 1.0
            else:
                cswap_matrix[state, state] = 1.0
        self.state_vector = cswap_matrix.dot(self.state_vector)
        return self

    # IMPLEMENTATION ESSENTIALS

    def __str__(self):
        result_str = ""
        for state in range(self.num_states):
            result_str += "{0:0>3b}".format(state) + " => {:.2f}".format(self.state_vector[state]) + "\n"
        return result_str[:-1]

    def plot_state_vector(self):
        plt.bar(range(self.num_states), np.absolute(self.state_vector), color='k')
        plt.title(str(self.num_qubits) + ' qubit register')
        plt.axis([0, self.num_states, 0.0, 1.0])
        plt.show()

    def save_plot(self, filename):
        plt.bar(range(self.num_states), np.absolute(self.state_vector), color='k')
        plt.title(str(self.num_qubits) + ' qubit register')
        plt.axis([0, self.num_states, 0.0, 1.0])
        plt.savefig("img/" + filename + ".pdf")

    def plot_alternative(self, save=None, filename=None):
        cols = 2 ** (self.num_qubits // 2)
        rows = 2 ** (self.num_qubits - (self.num_qubits // 2))

        x = []
        y = []
        colors = []

        for i in range(self.num_states):
            x.append(i % cols)
            y.append(i // cols)
            colors.append(np.absolute(self.state_vector[i]))

        plt.xlim(-0.5, cols-0.5)
        plt.ylim(-0.5, rows-0.5)

        plt.axes().set_aspect('equal')

        plt.scatter(x, y, s=2000, c=colors, linewidths=2, vmin=0, vmax=1, cmap=plt.get_cmap('jet'))

        if save is None:
            plt.show()
        else:
            plt.axis('off')
            plt.title('(' + filename + ')')

            fig = plt.gcf()
            fig.set_size_inches(cols, rows)
            fig.savefig("img/" + save + ".pdf", transparent=True, pad_inches=0)
