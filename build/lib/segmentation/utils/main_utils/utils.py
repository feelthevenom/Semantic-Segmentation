import os
import json
import matplotlib.pyplot as plt
import numpy as np


def save_metrics(metrics_data, file_path):
    """
    Save training, validation, or test metrics to a JSON file.

    Args:
        metrics_dict (dict): Dictionary containing metrics (IoU, Accuracy, etc.).
        file_path (str): Path to the JSON file where metrics should be stored.

    This function appends new metrics to the existing JSON file while keeping previous records.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing metrics if the file already exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append new metrics
    existing_data.append(metrics_data)

    # Custom function to convert non-serializable types to serializable ones.
    def default_converter(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        return o

    # Save back to the JSON file using the default converter.
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4, default=default_converter)

    print(f"Metrics saved to {file_path}")


def plot_metrics(metrics_file_path, save_dir):
    """
    Plot training and validation metrics over epochs and save the plots.

    Args:
        metrics_file_path (str): Path to the JSON file containing metrics.
        save_dir (str): Directory where plots should be saved.

    This function reads the metrics JSON file, extracts values, and plots separate graphs 
    for IoU, Accuracy, Precision, Recall, and F-score over all epochs.
    """

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load metrics from JSON file
    if not os.path.exists(metrics_file_path):
        print(f"Metrics file not found: {metrics_file_path}")
        return

    with open(metrics_file_path, "r") as file:
        try:
            metrics_data = json.load(file)
        except json.JSONDecodeError:
            print("Error reading metrics file!")
            return

    # Initialize lists for each metric
    epochs = []
    train_iou, valid_iou = [], []
    train_acc, valid_acc = [], []
    train_prec, valid_prec = [], []
    train_rec, valid_rec = [], []
    train_fscore, valid_fscore = [], []

    # Extract data from JSON
    for entry in metrics_data:
        if "epoch" in entry:
            epochs.append(entry["epoch"])
            train_iou.append(entry["train_metrics"].get("iou_score", 0))
            valid_iou.append(entry["valid_metrics"].get("iou_score", 0))

            train_acc.append(entry["train_metrics"].get("accuracy", 0))
            valid_acc.append(entry["valid_metrics"].get("accuracy", 0))

            train_prec.append(entry["train_metrics"].get("precision", 0))
            valid_prec.append(entry["valid_metrics"].get("precision", 0))

            train_rec.append(entry["train_metrics"].get("recall", 0))
            valid_rec.append(entry["valid_metrics"].get("recall", 0))

            train_fscore.append(entry["train_metrics"].get("fscore", 0))
            valid_fscore.append(entry["valid_metrics"].get("fscore", 0))

    # Function to plot graphs
    def save_plot(train_vals, valid_vals, ylabel, title, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_vals, label="Train", marker="o", linestyle="-")
        plt.plot(epochs, valid_vals, label="Validation", marker="o", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    # Generate and save plots
    save_plot(train_iou, valid_iou, "IoU Score", "IoU Over Epochs", "iou_plot.png")
    save_plot(train_acc, valid_acc, "Accuracy", "Accuracy Over Epochs", "accuracy_plot.png")
    save_plot(train_prec, valid_prec, "Precision", "Precision Over Epochs", "precision_plot.png")
    save_plot(train_rec, valid_rec, "Recall", "Recall Over Epochs", "recall_plot.png")
    save_plot(train_fscore, valid_fscore, "F1-Score", "F1-Score Over Epochs", "fscore_plot.png")

    print(f"Metrics plots saved in: {save_dir}")
