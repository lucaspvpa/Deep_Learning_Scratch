import numpy as np
from tensor import *

def randint(low=0, high=0, shape=(), requires_grad=False):
    data = np.random.randint(low, high, size=shape)
    return Tensor(data, requires_grad=requires_grad)

def randn(shape, requires_grad=False):
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)

def sum(a:Tensor, dim:int=-1, keepedims:bool=False):
    return a.sum(dim=dim, keepdims=keepedims)