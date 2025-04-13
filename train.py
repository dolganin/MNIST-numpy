import logging
import numpy as np
from tqdm import tqdm

from Utils.tools import setup_logging, batch_generator
from Utils.metrics import Precision, Recall, Accuracy
from Utils.tensor import Tensor

def train(model, optimizer, criterion, X_train, y_train, *, epochs=10, batch_size=32, validation_data=None):
    setup_logging()
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = Accuracy()

    # Метрики тренировки
    train_losses, train_accuracies, train_precisions, train_recalls = [], [], [], []

    # Метрики валидации (если есть)
    if validation_data:
        val_losses, val_accuracies, val_precisions, val_recalls = [], [], [], []

    num_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_precisions, batch_recalls, batch_accuracies = [], [], []

        progress_bar = tqdm(batch_generator(X_train, y_train, batch_size), desc=f"Epoch {epoch + 1}", total=num_batches)

        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = Tensor(X_batch)

            # Forward pass
            outputs = model(X_batch)
            loss_tensor = criterion(outputs, y_batch)
            batch_loss = loss_tensor.data.item()
            epoch_loss += batch_loss

            # Predictions
            predictions = np.argmax(outputs.data, axis=1)

            # Метрики
            batch_precision = precision_metric(y_batch, predictions) * 100
            batch_recall = recall_metric(y_batch, predictions) * 100
            batch_accuracy = accuracy_metric(y_batch, predictions) * 100

            batch_precisions.append(batch_precision)
            batch_recalls.append(batch_recall)
            batch_accuracies.append(batch_accuracy)

            # Backward pass
            grad_loss = criterion.backward()
            model.backward(grad_loss)

            # Обновление параметров
            params = model.parameters()
            grads = [p.grad for p in params]
            optimizer.step(grads)

            # Обнуляем градиенты
            for p in params:
                p.grad = np.zeros_like(p.grad)

            # Progress bar + logging
            progress_bar.set_postfix({
                "Loss": f"{batch_loss:.4f}",
                "Accuracy": f"{batch_accuracy:.2f}%",
                "Precision": f"{batch_precision:.2f}%",
                "Recall": f"{batch_recall:.2f}%"
            })
            logging.info(
                f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                f"Loss = {batch_loss:.4f}, Accuracy = {batch_accuracy:.2f}%, "
                f"Precision = {batch_precision:.2f}%, Recall = {batch_recall:.2f}%"
            )

        # Epoch summary
        epoch_accuracy = np.mean(batch_accuracies)
        epoch_precision = np.mean(batch_precisions)
        epoch_recall = np.mean(batch_recalls)

        train_losses.append(epoch_loss / num_batches)
        train_accuracies.append(epoch_accuracy)
        train_precisions.append(epoch_precision)
        train_recalls.append(epoch_recall)

        epoch_log_msg = (
            f"[Train] Epoch {epoch + 1} Summary: "
            f"Loss = {train_losses[-1]:.4f}, Accuracy = {epoch_accuracy:.2f}%, "
            f"Precision = {epoch_precision:.2f}%, Recall = {epoch_recall:.2f}%"
        )
        print(epoch_log_msg)
        logging.info(epoch_log_msg)

        # Валидация после эпохи
        if validation_data:
            X_val, y_val = validation_data
            val_loss, val_accuracy, val_precision, val_recall = test(
                model, criterion, X_val, y_val, batch_size=batch_size, epoch=epoch + 1
            )

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)

    if validation_data:
        return (train_losses, train_accuracies, train_precisions, train_recalls), \
               (val_losses, val_accuracies, val_precisions, val_recalls)
    else:
        return train_losses, train_accuracies, train_precisions, train_recalls


def test(model, criterion, X_test, y_test, *, batch_size=32, epoch=None):
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = Accuracy()

    test_loss = 0.0
    batch_precisions, batch_recalls, batch_accuracies = [], [], []

    num_batches = int(np.ceil(len(X_test) / batch_size))
    desc = f"Validation Epoch {epoch}" if epoch else "Testing"
    progress_bar = tqdm(batch_generator(X_test, y_test, batch_size), desc=desc, total=num_batches)

    for X_batch, y_batch in progress_bar:
        X_batch = Tensor(X_batch)

        # Forward pass
        outputs = model(X_batch)
        loss_tensor = criterion(outputs, y_batch)
        batch_loss = loss_tensor.data.item()
        test_loss += batch_loss

        # Predictions and metrics
        predictions = np.argmax(outputs.data, axis=1)

        batch_precision = precision_metric(y_batch, predictions) * 100
        batch_recall = recall_metric(y_batch, predictions) * 100
        batch_accuracy = accuracy_metric(y_batch, predictions) * 100

        batch_precisions.append(batch_precision)
        batch_recalls.append(batch_recall)
        batch_accuracies.append(batch_accuracy)

        # Progress bar
        progress_bar.set_postfix({
            "Loss": f"{batch_loss:.4f}",
            "Accuracy": f"{batch_accuracy:.2f}%",
            "Precision": f"{batch_precision:.2f}%",
            "Recall": f"{batch_recall:.2f}%"
        })

    # Final metrics
    avg_loss = test_loss / num_batches
    avg_accuracy = np.mean(batch_accuracies)
    avg_precision = np.mean(batch_precisions)
    avg_recall = np.mean(batch_recalls)

    summary_msg = (
        f"[Test] Epoch {epoch if epoch else '-'} Summary: "
        f"Loss = {avg_loss:.4f}, "
        f"Accuracy = {avg_accuracy:.2f}%, "
        f"Precision = {avg_precision:.2f}%, "
        f"Recall = {avg_recall:.2f}%"
    )
    print(summary_msg)
    logging.info(summary_msg)

    return avg_loss, avg_accuracy, avg_precision, avg_recall
