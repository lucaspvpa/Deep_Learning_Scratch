import numpy as np
from tensor import *
from utils import *

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
    def __init__(self, n_neurons=0, n_input=None):
        self.n_neurons = None
        if n_input:
            self.weights = randn((n_input[-1], n_neurons), requires_grad=True)
            self.bias = randn((1, n_neurons), requires_grad=True)
        else:
            self.n_neurons = n_neurons

    def __call__(self, input):
        input = tensor(input)
        if self.n_neurons:
            self.weights = randn((input.shape[-1], self.n_neurons), requires_grad=True)
            self.bias = randn((1, self.n_neurons), requires_grad=True)
        self.output = (input @ self.weights) + self.bias
        return self.output

class ReLU():
    def __call__(self, input):
        mask = Tensor(np.where(input._data < 0, 0, 1))
        input = input * mask
        return input