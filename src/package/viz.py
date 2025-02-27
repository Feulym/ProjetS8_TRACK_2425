import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List
from sklearn.metrics import confusion_matrix


def plot_metrics(metrics_file: str = "metrics.csv", show_plot: bool = False) -> None:
    """
    Plot the training and validation loss and accuracy over epochs from a CSV metrics file.
    
    Args:
        metrics_file (str): Path to the CSV file containing the metrics.
        save_path (str): Path to save the plot as a PNG file.
        show_plot (bool): If True, the plot will be displayed. If False, it will only be saved.
    """
    save_path = os.path.splitext(metrics_file)[0] + ".png"

    epochs = []
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []

    with open(metrics_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_accuracy.append(float(row["train_accuracy"]))
            val_loss.append(float(row["val_loss"]))
            val_accuracy.append(float(row["val_accuracy"]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Training and Validation Accuracy", fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_cm(evaluations: Tuple[np.ndarray, np.ndarray], class_names: List[str] = None,
               save_path: str = "confusion_matrix.png", show_figure: bool = False) -> None:
    y_true, y_pred = evaluations
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100   # TODO: Check: cm.sum(axis=1, keepdims=True)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    plt.colorbar(label='Percentage')
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha='center', va='center',
                     color="black" if cm_percent[i, j] < 50 else "white")

    plt.savefig(save_path, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    
    if show_figure:
        plt.show()
    else:
        plt.close()