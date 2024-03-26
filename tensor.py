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
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def __add__(self, other):
        """ __add__ = self + other"""
        op = Add()
        return op.forward(self, tensor(other))
    
    def __radd__(self, other):
        """__radd__ = other + self"""
        op = Add()
        return op.forward(self, tensor(other))
    
    def __iadd__(self, other):
        """ __iadd__ = self += other """
        op = Add()
        return op.forward(self, tensor(other))
    
    def __sub__(self, other):
        """ __sub__ = self - other"""
        return self + -other
    
    def __rsub__(self, other):
        """ __rsub__ = other - self"""
        return other - self
    
    def __isub__(self, other):
        """ __isub__ = self -= other"""
        return self + -other
    
    def __neg__(self):
        """ __neg__ = self = -self"""
        op = Neg()
        return op.forward(self)
    
    def __mul__(self, other):
        """ __mul__ = self * other"""
        op = Mul()
        return op.forward(self, tensor(other))
    
    def __rmul__(self, other):
        """ __rmul__ = other * self"""
        op = Mul()
        return op.forward(self, tensor(other))
    
    def __imul__(self, other):
        """ __imul__ = self *= other"""
        op = Mul()
        return op.forward(self, tensor(other))
    
    def __matmul__(self, other):
        """ __matmul__ = self @ other"""
        op = MatMul()
        return op.forward(self, tensor(other))
    
class Add:
    def forward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
        data = tensor_a._data + tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        tensor_b.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z

    def backward(self, dz, z):
        tensor_a, tensor_b = self.cache
        if tensor_a.requires_grad:
            da = dz
            grad_dim = len(dz.shape)
            in_dim = len(tensor_a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(tensor_a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            tensor_a.backward(da, z)
            
        if tensor_b.requires_grad:
            db = dz
            grad_dim = len(dz.shape)
            in_dim = len(tensor_b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(tensor_b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            tensor_b.backward(db, z)

class MatMul():
    def forward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
        data = tensor_a._data @ tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        tensor_b.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z

    def backward(self, dz, z):
        tensor_a, tensor_b = self.cache
        if tensor_a.requires_grad:
            da = dz @ tensor_b._data.swapaxes(-1, -2)
            grad_dim = len(da.shape)
            in_dim = len(tensor_a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            tensor_a.backward(da, z)
        
        if tensor_b.requires_grad:
            db =  tensor_a._data.swapaxes(-1, -2) @ dz
            grad_dim = len(db.shape)
            in_dim = len(tensor_b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            tensor_b.backward(db, z)

class Neg():
    def forward(self, tensor_a):
        requires_grad = tensor_a.requires_grad
        data = - tensor_a._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        self.cache = tensor_a
        return z
    
    def backward(self, dz, z):
        tensor_a = self.cache
        if tensor_a.requires_grad:
            da = -dz
            tensor_a.backward(da, z)

class Mul():
    def forward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad or tensor_b.requires_grad
        data = tensor_a._data * tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        tensor_b.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z
    
    def backward(self, dz, z):
        tensor_a, tensor_b = self.cache
        if tensor_a.requires_grad:
            da = dz * tensor_b._data
            grad_dim = len(dz.shape)
            in_dim = len(tensor_a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            for n, dim in enumerate(tensor_a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            tensor_a.backward(da, z)
        
        if tensor_b.requires_grad:
            db = dz * tensor_a._data
            grad_dim = len(dz.shape)
            in_dim = len(tensor_b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            for n, dim in enumerate(tensor_b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            tensor_b.backward(db, z)

               
def randint(low=0, high=0, shape=(), requires_grad=False):
    data = np.random.randint(low, high, size=shape)
    return Tensor(data, requires_grad=requires_grad)

def randn(shape, requires_grad=False):
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)
    
def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)