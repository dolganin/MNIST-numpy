import abc
import numpy as np

from Utils.tensor import Tensor


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass
    
    @property
    @abc.abstractmethod
    def params(self):
        return []