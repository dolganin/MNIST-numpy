import os
import pandas as pd

from Utils.tools import yaml_reader, plot_metrics, DataLoader
from Model.model import MnistClassifier
from Optimizer.adam import AdamOptimizer
from Loss.ce import CrossEntropyLoss
from train import train
from Augmentations.augs import Compose, RandomHorizontalFlip, RandomRotation, AddGaussianNoise, ToFloat, Normalize

def main():
    # Читаем конфиг
    config = yaml_reader()

    # Параметры данных
    traindir = config["dataset_parameters"]["traindir"]
    testdir = config["dataset_parameters"]["testdir"]

    # Гиперпараметры тренировки
    learning_rate = config["training_parameters"]["learning_rate"]
    epochs = config["training_parameters"]["num_epochs"]
    batch_size = config["training_parameters"]["batch_size"]

    # Параметры модели
    dropout_rate = config["model_parameters"]["dropout_rate"]
    weight_decay = config["model_parameters"]["weight_decay"]
    beta1 = config["model_parameters"]["beta1"]
    beta2 = config["model_parameters"]["beta2"]
    eps = config["model_parameters"]["eps"]

    # Параметры вывода
    graphics_dir = config["output_parameters"]["out_graphics_directory"]
    model_dir = config["output_parameters"]["out_model_directory"]

    # Загружаем данные
    data_train = pd.read_csv(traindir).to_numpy()
    data_test = pd.read_csv(testdir).to_numpy()

    # Аугментации
    train_transforms = Compose([
        ToFloat(),
        Normalize(mean=0.1307, std=0.3081),
        RandomHorizontalFlip(prob=0.1),
        AddGaussianNoise(mean=0.0, std=0.05),
    ])
    test_transforms = Compose([
    ToFloat(),
    Normalize(mean=0.1307, std=0.3081)])

    # DataLoader'ы
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, augmentations=train_transforms)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, augmentations=test_transforms)

    # Инициализация модели, оптимизатора и функции потерь
    model = MnistClassifier()
    optimizer = AdamOptimizer(
        model.parameters(),
        lr=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=float(eps),
        weight_decay=weight_decay
    )
    criterion = CrossEntropyLoss()

    # Тренировка модели + тест
    train_metrics, test_metrics = train(
        model, optimizer, criterion,
        train_loader,
        validation_data=test_loader,
        epochs=epochs
    )

    # Сохраняем графики
    plot_metrics(train_metrics, test_metrics, output_dir=graphics_dir)

    # Сохраняем веса модели
    model.save_weights(model_dir)

if __name__ == "__main__":
    main()
