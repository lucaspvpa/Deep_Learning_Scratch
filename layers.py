import numpy as np

class Dense():
    """
        Classe que define a camada Dense (hidden layer):
            forward: aplica a equação (Input x Weight) + bias
            backward: aplica a derivada parcial pela regra da cadeia,
            necessário para implementação do algoritmo de backpropagation

        Attributes:
            weights: é um array multidimensional que contem os pesos da camada
            a ser multiplicada (dot product) pela entrada
            bias: array com o mesmo tamanho do número de neurônios, serve como
            auxiliar para generalização do modelo
    """
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def forward(self, input):
        self.output = (input @ self.weights) + self.bias

    def backward(self):
        pass