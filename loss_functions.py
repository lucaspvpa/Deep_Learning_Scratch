import numpy as np
from tensor import *
from utils import *

class MSE():
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        y_pred = tensor(y_pred)
        y_true = tensor(y_true)
        y_pred_reshape = y_pred.reshape(*y_true.shape)

        self.mse = (1 / y_true.shape[1]) * sum((y_true - y_pred_reshape)**2)
        return self.mse