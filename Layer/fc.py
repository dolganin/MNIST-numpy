class FC(Layer):
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features) * 0.1, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.last_input = None
        
    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.data
        return Tensor(x.data.dot(self.weights.data) + self.bias.data)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Градиенты для весов и смещений
        self.weights.grad = self.last_input.T.dot(grad)
        self.bias.grad = np.sum(grad, axis=0)
        # Градиент по входу слоя
        return grad.dot(self.weights.data.T)
    
    @property
    def params(self):
        return [self.weights, self.bias]