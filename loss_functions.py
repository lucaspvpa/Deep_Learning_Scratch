import numpy as np
from tensor import *
from utils import *

class MSE():
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        y_pred = tensor(y_pred)
        y_true = tensor(y_true)
        self.mse = (1 / y_pred.shape[1]) * sum((y_true - y_pred)**2)
        return self.mse