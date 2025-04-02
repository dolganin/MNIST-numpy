class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size[0]
        self.last_input = None
        self.max_indices = None
        
    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.data
        batch_size, channels, H, W = x.data.shape
        kH, kW = self.kernel_size
        
        out_H = (H - kH) // self.stride + 1
        out_W = (W - kW) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_H, out_W))
        max_indices = np.zeros((batch_size, channels, out_H, out_W, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        h_end = h_start + kH
                        w_start = j * self.stride
                        w_end = w_start + kW
                        
                        window = x.data[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        output[b, c, i, j] = max_val
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])
        
        self.max_indices = max_indices
        return Tensor(output)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        dx = np.zeros_like(self.last_input)
        batch_size, channels, out_H, out_W = grad.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h, w = self.max_indices[b, c, i, j]
                        dx[b, c, h, w] = grad[b, c, i, j]
        return dx
    
    @property
    def params(self):
        return []