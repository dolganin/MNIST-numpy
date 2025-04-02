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
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging():
    logging.basicConfig(filename='training.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')