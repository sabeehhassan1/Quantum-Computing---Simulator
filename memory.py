import numpy as np


class QuantumMemory(object):
    def __init__(self, addressQubits, valueQubits): 
        self.addressQubits = addressQubits
        self.valueQubits = valueQubits

        self.addressStates = 2 ** addressQubits
        self.valueStates = 2 ** valueQubits
        
        self.storage = np.zeros((self.addressStates, self.valueStates), dtype=complex)

    def read(self, address): 
        result = np.zeros((self.valueStates,), dtype=complex)
        for i in range(address.shape[0]):
            result += address[i] ** 2 * self.storage[i, :]
        return result

    def write(self, address, value):
        assert address < self.addressStates
        self.storage[address,:] = value
