import numpy as np

class MSE():
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        self.mse = (1 / len(y_pred)) * np.sum((y_true - y_pred)**2)

    def backward(self):
        pass