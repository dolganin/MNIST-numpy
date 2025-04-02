class Flatten(Layer):
    def forward(self, x: Tensor) -> Tensor:
        self.input_shape = x.data.shape
        return Tensor(x.data.reshape(x.data.shape[0], -1))
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.input_shape)
    
    @property
    def params(self):
        return []