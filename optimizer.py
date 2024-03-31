import numpy as np

class SGD():
    def __init__(self, parameters, lr, momentum=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        for parameter in self.parameters:
            parameter.m = np.zeros(parameter.shape)

    def step(self):
        for parameter in self.parameters:
            parameter.m = (self.momentum * parameter.m) - (self.lr * parameter.grad)
            parameter._data = parameter._data + parameter.m
        
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()