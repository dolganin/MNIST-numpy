class MnistClassifier:
    def __init__(self):
        # Свёрточный слой: входной канал=1, выход=8, ядро 3x3, stride=1, padding=1 (для сохранения размера)
        self.conv = Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = ReLU()
        # MaxPool с ядром 2x2 и stride=2 (уменьшает размер с 28 до 14)
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.flatten = Flatten()
        # Полносвязный слой: 8*14*14 входов -> 10 классов
        self.fc = FC(8 * 14 * 14, 10)
        self.layers = [self.conv, self.relu, self.pool, self.flatten, self.fc]
    
    def __call__(self, x: Tensor) -> Tensor:
        out = self.conv.forward(x)
        out = self.relu.forward(out)
        out = self.pool.forward(out)
        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        return out
    
    def backward(self, grad: np.ndarray):
        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.conv.backward(grad)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params)
        return params