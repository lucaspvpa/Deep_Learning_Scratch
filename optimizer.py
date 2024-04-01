import numpy as np

class SGD():
    def __init__(self, parameters, lr, momentum=0, nesterov=False, weigh_decay=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weigh_decay = weigh_decay
        for parameter in self.parameters:
            parameter.m = np.zeros(parameter.shape)

    def step(self):
        for parameter in self.parameters:
            gt = parameter.grad
            if self.weigh_decay != 0:
                gt = gt + (self.weigh_decay * parameter._data)
            if self.momentum != 0:
                parameter.m = (self.momentum * parameter.m) - gt
                if self.nesterov:
                    gt = gt + (self.momentum * parameter.m)
            parameter._data = parameter._data - (self.lr * gt)
        
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


class RMSProp:
    def __init__(self, parameters, lr, decay_rate=0.9, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        for parameter in self.parameters:
            parameter.vt = np.zeros(parameter.shape)

    def step(self):
        for parameter in self.parameters:
            gt = parameter.grad
            parameter.vt = (self.decay_rate * parameter.vt) + (1 - self.decay_rate) * (gt**2)
            parameter._data = parameter._data - (self.lr * gt / ((parameter.vt**0.5) + self.eps)) 

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

class Adam():
    pass