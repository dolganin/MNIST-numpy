import numpy as np

from Layer.base_layer import Layer
from Utils.tensor import Tensor

class Conv2d(Layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, mode='naive'):
        self.weights = Tensor(
            np.random.randn(out_channel, in_channel, *kernel_size) * 0.1,
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channel), requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.mode = mode  # 'naive' или 'im2col'

        self.last_input = None
        self.last_cols = None

    def im2col(self, x, kH, kW, stride, padding):
        batch_size, in_channel, H, W = x.shape
        out_H = (H + 2 * padding - kH) // stride + 1
        out_W = (W + 2 * padding - kW) // stride + 1

        # Padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        cols = np.zeros((batch_size, in_channel, kH, kW, out_H, out_W))
        for y in range(kH):
            y_max = y + stride * out_H
            for x in range(kW):
                x_max = x + stride * out_W
                cols[:, :, y, x, :, :] = x_padded[:, :, y:y_max:stride, x:x_max:stride]

        cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(in_channel * kH * kW, -1)
        return cols

    def col2im(self, cols, x_shape, kH, kW, stride, padding):
        batch_size, in_channel, H, W = x_shape
        out_H = (H + 2 * padding - kH) // stride + 1
        out_W = (W + 2 * padding - kW) // stride + 1

        cols_reshaped = cols.reshape(in_channel, kH, kW, batch_size, out_H, out_W).transpose(3, 0, 1, 2, 4, 5)
        x_padded = np.zeros((batch_size, in_channel, H + 2 * padding, W + 2 * padding))

        for y in range(kH):
            y_max = y + stride * out_H
            for x in range(kW):
                x_max = x + stride * out_W
                x_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]

        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.data
        batch_size, in_channel, H, W = x.data.shape
        out_channel, _, kH, kW = self.weights.data.shape

        out_H = (H - kH + 2 * self.padding) // self.stride + 1
        out_W = (W - kW + 2 * self.padding) // self.stride + 1

        if self.mode == 'naive':
            # Старая версия
            padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
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

        elif self.mode == 'im2col':
            cols = self.im2col(x.data, kH, kW, self.stride, self.padding)
            self.last_cols = cols

            reshaped_weights = self.weights.data.reshape(out_channel, -1)
            output = reshaped_weights @ cols + self.bias.data[:, None]
            output = output.reshape(out_channel, batch_size, out_H, out_W).transpose(1, 0, 2, 3)
            return Tensor(output)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size, in_channel, H, W = self.last_input.shape
        out_channel, _, kH, kW = self.weights.data.shape

        if self.mode == 'naive':
            # Старая версия
            padded_x = np.pad(self.last_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dx = np.zeros_like(padded_x)

            self.weights.grad = np.zeros_like(self.weights.data)
            self.bias.grad = np.zeros_like(self.bias.data)

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

            if self.padding != 0:
                dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
            return dx

        elif self.mode == 'im2col':
            # Градиенты
            dout = grad.transpose(1, 0, 2, 3).reshape(out_channel, -1)

            self.bias.grad = np.sum(dout, axis=1)
            self.weights.grad = (dout @ self.last_cols.T).reshape(self.weights.data.shape)

            dcols = self.weights.data.reshape(out_channel, -1).T @ dout
            dx = self.col2im(dcols, self.last_input.shape, kH, kW, self.stride, self.padding)
            return dx

        else:
            raise ValueError(f"Unknown mode {self.mode}")

    @property
    def params(self):
        return [self.weights, self.bias]
