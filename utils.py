import numpy as np
from tensor import *

def randint(low=0, high=0, shape=(), requires_grad=False):
    data = np.random.randint(low, high, size=shape)
    return Tensor(data, requires_grad=requires_grad)

def randn(shape, requires_grad=False):
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)

def sum(tensor_a:Tensor, dim:int=-1, keepedims:bool=False):
    return tensor_a.sum(dim=dim, keepdims=keepedims)

def max(tensor_a: Tensor, dim: int=-1, keepdims: bool=False):
    return tensor_a.max(dim=dim, keepdims=keepdims)