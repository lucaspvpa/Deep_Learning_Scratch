import numpy as np

class Tensor():
    def __init__(self, data, requires_grad=False, operation = None):
        self._data = np.array(data)
        self.requires_grad = requires_grad
        self.children = []
        self.operation = operation
        self.shape = self._data.shape
        if self.requires_grad:
            self.grad = np.zeros_like(data)


    def __repr__(self):
        return f"({self._data}, requires_grad = {self.requires_grad})"
    
    def backward(self, grad=None, z=None):
        if not self.requires_grad:
            return "O tensor tem o parâmetro requires_grad como Falso"
        
        if grad is None:
            grad = np.ones_like(self._data)
        
        self.grad += grad

        if z is not None:
            print(z)
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def __add__(self, other):
        """ __add__ = self + other"""
        op = Add()
        return op.foward(self, tensor(other))
    
    def __radd__(self, other):
        """__radd__ = other + self"""
        op = Add()
        return op.foward(self, tensor(other))
    
    def __iadd__(self, other):
        """ __iadd__ = self += other """
        op = Add()
        return op.foward(self, tensor(other))
    
    def __matmul__(self, other):
        op = MatMul()
        return op.foward(self, tensor(other))
    
class Add:
    def foward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
        data = tensor_a._data + tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        tensor_b.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z

    def backward(self, dz, z):
        a, b = self.cache
        if a.requires_grad:
            da = dz
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)
            
        if b.requires_grad:
            db = dz
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)

class MatMul():
    def foward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
        data = tensor_a._data @ tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        tensor_b.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z

    def backward(self, dz, z):
        a, b = self.cache
        if a.requires_grad:
            da = dz @ b._data.swapaxes(-1, -2)
            grad_dim = len(da.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            a.backward(da, z)
        
        if b.requires_grad:
            db =  a._data.swapaxes(-1, -2) @ dz
            grad_dim = len(db.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            b.backward(db, z)

def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)