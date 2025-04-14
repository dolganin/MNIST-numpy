import yaml
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

from Augmentations.augs import Compose, RandomHorizontalFlip, RandomRotation, AddGaussianNoise, ToFloat

class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True, augment=False, augmentations=None):
        self.X, self.y = self.preprocess_data(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Если передали внешние аугментации — берём их, иначе дефолтные
        if augmentations is not None:
            self.augmentations = augmentations
        elif augment:
            from Augmentations.augs import Compose, RandomHorizontalFlip, RandomRotation, AddGaussianNoise, ToFloat
            self.augmentations = Compose([
                ToFloat(),
                RandomHorizontalFlip(prob=0.5),
                RandomRotation(angles=[0, 90, 180, 270]),
                AddGaussianNoise(mean=0.0, std=0.05),
            ])
        else:
            self.augmentations = None

    @staticmethod
    def preprocess_data(data):
        X = data[:, 1:].astype(np.float32).reshape(-1, 1, 28, 28)
        y = data[:, 0].astype(int)
        return X, y

    def __iter__(self):
        indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.X), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.X))
            batch_idx = indices[start_idx:end_idx]

            X_batch = self.X[batch_idx]
            y_batch = self.y[batch_idx]

            if self.augmentations:
                X_batch = np.array([self.augmentations(x) for x in X_batch])

            yield X_batch, y_batch

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))



# =======================================
# Старый генератор можно оставить для отладки, но в продакшне DataLoader лучше
def batch_generator(X, y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]


def yaml_reader():
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging():
    logging.basicConfig(filename='training.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def plot_metrics(train_metrics, val_metrics, output_dir):
    train_losses, train_accuracies, *_ = train_metrics
    val_losses, val_accuracies, *_ = val_metrics

    steps_train = range(1, len(train_losses) + 1)
    steps_val = range(1, len(val_losses) + 1)

    plt.figure(figsize=(14, 6))

    # === LOSS ===
    plt.subplot(1, 2, 1)
    plt.plot(steps_train, train_losses, label='Train Loss', color='blue')
    plt.plot(steps_val, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Batch Step')
    plt.ylabel('Loss')
    plt.title('Loss per Batch')
    plt.legend()

    # === ACCURACY ===
    plt.subplot(1, 2, 2)
    plt.plot(steps_train, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(steps_val, val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Batch Step')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Batch')
    plt.legend()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_validation_metrics.png")
    plt.savefig(output_path)
    print(f"Graph saved to: {output_path}")

    plt.show()


