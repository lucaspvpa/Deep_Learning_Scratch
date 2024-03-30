class SGD():
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter._data = parameter._data - (self.lr * parameter.grad)
        
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()