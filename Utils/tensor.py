class Tensor:
    def __init__(self, value: np.ndarray, requires_grad=False):
        self.data = value
        self.grad = np.zeros_like(value) if requires_grad else None
        self.requires_grad = requires_grad
        
    def backward(self, grad=None):
        if self.requires_grad:
            if grad is None:
                grad = np.ones_like(self.data)
            self.grad += grad
            
    def __repr__(self):
        return f"Tensor({self.data.shape}, grad={self.grad is not None})"