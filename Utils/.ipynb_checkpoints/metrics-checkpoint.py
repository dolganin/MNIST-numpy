class Precision:
    def __call__(self, y_true, y_pred):
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / (predicted_positive + 1e-8)

class Recall:
    def __call__(self, y_true, y_pred):
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / (actual_positive + 1e-8)