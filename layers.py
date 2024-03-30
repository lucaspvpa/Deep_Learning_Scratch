import numpy as np
from tensor import *
from utils import *

class Module():
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                params += param.parameters()
            elif isinstance(param, Tensor):
                if param.requires_grad:
                    params.append(param)
        return params
    
    def train(self):
        self.mode = 'train'
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.train()
    
    def eval(self):
        self.mode = 'eval'
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.eval()

class Dense(Module):
    def __init__(self, n_neurons=None, n_input=None, bias=True):
        super().__init__()
        self.has_bias = bias
        self.weights = Tensor(np.random.randn(n_input, n_neurons), requires_grad=True)
        if self.has_bias:
            self.bias = Tensor(np.zeros(n_neurons), requires_grad=True)

    def forward(self, input):
        input = tensor(input)
        self.output = input @ self.weights 
        if self.has_bias:
            self.output += self.bias
        return self.output

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        mask = Tensor(np.where(input._data < 0, 0, 1))
        input = input * mask
        return input