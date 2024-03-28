from tensor import *
from layers import *
from loss_functions import *
from utils import *

class Model():
    def __init__(self):
        self.list_layer = []

    def add_layers(self, layer):
        self.list_layer.append(layer)

    def train(self, x_train, y_train):
        output = x_train
        for layer in self.list_layer:
            output = layer(output)
        output.backward()
