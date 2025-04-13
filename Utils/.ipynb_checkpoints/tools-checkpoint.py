import yaml
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

def batch_generator(X, y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]


def preprocess_data(data):
    X = data[:, 1:].astype(np.float32) / 255.0  # Нормализация изображений
    X = X.reshape(-1, 1, 28, 28)  # Преобразуем в тензоры (batch, channels, height, width)
    y = data[:, 0].astype(int)  # Метки классов
    return X, y

def yaml_reader():
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging():
    logging.basicConfig(filename='training.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')





def plot_metrics(train_metrics, test_metrics, output_dir):
    train_losses, train_accuracies, train_precisions, train_recalls = train_metrics
    test_losses, test_accuracies, test_precisions, test_recalls = test_metrics

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, test_precisions, label='Test Precision')
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, test_recalls, label='Test Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics (%)')
    plt.title('Metrics per Epoch')
    plt.legend()

    plt.tight_layout()

    # Проверяем, существует ли директория для графиков
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем в PNG
    output_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(output_path)
    print(f"График сохранён в: {output_path}")

    plt.show()
