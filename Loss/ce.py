from Utils.tensor import Tensor
import numpy as np

class CrossEntropyLoss:
    def __call__(self, logits: Tensor, target: np.ndarray) -> Tensor:
        self.logits = logits.data
        self.target = target
        # Вычисление softmax
        exp_logits = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        m = target.shape[0]
        loss = -np.sum(np.log(self.probs[range(m), target])) / m
        return Tensor(np.array(loss))
    
    def backward(self):
        m = self.target.shape[0]
        grad = self.probs.copy()
        grad[range(m), self.target] -= 1
        grad /= m
        return grad