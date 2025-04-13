import os
import numpy as np

from Utils.tensor import Tensor
from Layer.conv_2d import Conv2d
from Layer.activations import ReLU
from Layer.max_pool_2d import MaxPool2d
from Layer.flatten import Flatten
from Layer.fc import FC

class MnistClassifier:
    def __init__(self):
        # Слои модели
        self.layers = [
            Conv2d(in_channel=1, out_channel=8, kernel_size=(3, 3), stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),
            Flatten(),
            FC(in_features=8 * 14 * 14, out_features=10)
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params)
        return params

    def save_weights(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        weights = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_idx, param in enumerate(layer.params):
                    key = f"layer{idx}_param{param_idx}"
                    weights[key] = param.data

        output_path = os.path.join(output_dir, 'model_weights.npz')
        np.savez(output_path, **weights)
        print(f"Веса модели сохранены в: {output_path}")

    def load_weights(self, weights_path):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Файл весов не найден по пути: {weights_path}")

        weights = np.load(weights_path)
        print(f"Загрузка весов модели из: {weights_path}")

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_idx, param in enumerate(layer.params):
                    key = f"layer{idx}_param{param_idx}"
                    if key in weights:
                        param.data = weights[key]
                    else:
                        print(f"Предупреждение: ключ {key} не найден в файле весов.")
