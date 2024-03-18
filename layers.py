import numpy as np

class Dense():
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def forward(self, input):
        self.output = (input @ self.weights) + self.bias

    def backward(self):
        pass