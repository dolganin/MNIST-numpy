import numpy as np

from Layer.base_layer import Layer
from Utils.tensor import Tensor

class ReLU(Layer):
    def __init__(self):
        self.last_input = None
        
    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.data
        return Tensor(np.maximum(0, x.data))
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.last_input > 0)
    
    @property
    def params(self):
        return []