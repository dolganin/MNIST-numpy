import logging
import numpy as np
from tqdm import tqdm

from Utils.tools import setup_logging
from Utils.metrics import Precision, Recall, Accuracy
from Utils.tensor import Tensor

def train(model, optimizer, criterion, train_loader, *, validation_data=None, epochs=1, validate_every=300):
    setup_logging()
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = Accuracy()

    train_losses, train_accuracies, train_precisions, train_recalls = [], [], [], []
    val_losses, val_accuracies, val_precisions, val_recalls = [], [], [], []

    global_step = 0

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", total=len(train_loader))

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            global_step += 1
            X_batch = Tensor(X_batch)

            outputs = model(X_batch)
            loss_tensor = criterion(outputs, y_batch)
            batch_loss = loss_tensor.data.item()

            predictions = np.argmax(outputs.data, axis=1)

            batch_precision = precision_metric(y_batch, predictions) * 100
            batch_recall = recall_metric(y_batch, predictions) * 100
            batch_accuracy = accuracy_metric(y_batch, predictions) * 100

            train_losses.append(batch_loss)
            train_accuracies.append(batch_accuracy)
            train_precisions.append(batch_precision)
            train_recalls.append(batch_recall)

            grad_loss = criterion.backward()
            model.backward(grad_loss)

            params = model.parameters()
            grads = [p.grad for p in params]
            optimizer.step(grads)

            for p in params:
                p.grad = np.zeros_like(p.grad)

            progress_bar.set_postfix({
                "Loss": f"{batch_loss:.4f}",
                "Accuracy": f"{batch_accuracy:.2f}%",
                "Precision": f"{batch_precision:.2f}%",
                "Recall": f"{batch_recall:.2f}%"
            })

            # Валидация каждые validate_every батчей
            if validation_data and global_step % validate_every == 0:
                val_loss, val_accuracy, val_precision, val_recall = test(
                    model, criterion, validation_data
                )

                # Учитываем, что test() возвращает списки по батчам
                val_losses.extend(val_loss)
                val_accuracies.extend(val_accuracy)
                val_precisions.extend(val_precision)
                val_recalls.extend(val_recall)

    return (train_losses, train_accuracies, train_precisions, train_recalls), \
           (val_losses, val_accuracies, val_precisions, val_recalls)

def test(model, criterion, test_loader):
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = Accuracy()

    batch_losses, batch_precisions, batch_recalls, batch_accuracies = [], [], [], []

    progress_bar = tqdm(test_loader, desc="Validation", total=len(test_loader))

    for X_batch, y_batch in progress_bar:
        X_batch = Tensor(X_batch)

        outputs = model(X_batch)
        loss_tensor = criterion(outputs, y_batch)
        batch_loss = loss_tensor.data.item()

        predictions = np.argmax(outputs.data, axis=1)

        batch_precision = precision_metric(y_batch, predictions) * 100
        batch_recall = recall_metric(y_batch, predictions) * 100
        batch_accuracy = accuracy_metric(y_batch, predictions) * 100

        batch_losses.append(batch_loss)
        batch_precisions.append(batch_precision)
        batch_recalls.append(batch_recall)
        batch_accuracies.append(batch_accuracy)

        progress_bar.set_postfix({
            "Loss": f"{batch_loss:.4f}",
            "Accuracy": f"{batch_accuracy:.2f}%",
            "Precision": f"{batch_precision:.2f}%",
            "Recall": f"{batch_recall:.2f}%"
        })

    return batch_losses, batch_accuracies, batch_precisions, batch_recalls


