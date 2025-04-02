class Conv2d(Layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        self.weights = Tensor(
            np.random.randn(out_channel, in_channel, *kernel_size) * 0.1,
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channel), requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.last_input = None
        
    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.data
        batch_size, in_channel, H, W = x.data.shape
        out_channel, _, kH, kW = self.weights.data.shape
        
        # Вычисление выходных размеров
        out_H = (H - kH + 2 * self.padding) // self.stride + 1
        out_W = (W - kW + 2 * self.padding) // self.stride + 1
        
        # Добавление паддинга
        padded = np.pad(x.data, 
                        ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                        mode='constant')
        
        output = np.zeros((batch_size, out_channel, out_H, out_W))
        for b in range(batch_size):
            for c in range(out_channel):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        h_end = h_start + kH
                        w_start = j * self.stride
                        w_end = w_start + kW
                        
                        window = padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.sum(window * self.weights.data[c]) + self.bias.data[c]
        return Tensor(output)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.last_input
        batch_size, in_channel, H, W = x.shape
        out_channel, _, kH, kW = self.weights.data.shape
        
        # Обнуление градиентов
        self.weights.grad = np.zeros_like(self.weights.data)
        self.bias.grad = np.zeros_like(self.bias.data)
        
        # Паддинг для градиентов
        padded_x = np.pad(x, 
                          ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')
        padded_grad = np.pad(grad, 
                             ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                             mode='constant')
        dx = np.zeros_like(padded_x)
        batch_size, _, out_H, out_W = grad.shape
        
        for b in range(batch_size):
            for c in range(out_channel):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        h_end = h_start + kH
                        w_start = j * self.stride
                        w_end = w_start + kW
                        
                        window = padded_x[b, :, h_start:h_end, w_start:w_end]
                        self.weights.grad[c] += window * grad[b, c, i, j]
                        dx[b, :, h_start:h_end, w_start:w_end] += self.weights.data[c] * grad[b, c, i, j]
                        
                self.bias.grad[c] += np.sum(grad[b, c])
        
        # Убираем паддинг из dx
        if self.padding != 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx
    
    @property
    def params(self):
        return [self.weights, self.bias]