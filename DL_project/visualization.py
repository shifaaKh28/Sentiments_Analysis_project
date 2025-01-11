import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

"""
Module for visualizing dataset distributions and training metrics.

This module includes:
- `plot_sentiment_distribution`: Visualizes the distribution of sentiments in the dataset.
- `plot_training_metrics`: Plots training and validation metrics over epochs.

Dependencies:
- `seaborn`: For creating visually appealing plots.
- `matplotlib.pyplot`: For creating visualizations and configuring plot properties.
- `matplotlib.font_manager`: For font customization in plots.
"""

def plot_sentiment_distribution(data, class_names):
    """
    Visualize the distribution of sentiments in the dataset.

    Args:
        data (DataFrame): The preprocessed dataset. Must contain a 'Sentiment' column.
        class_names (list): List of sentiment class names for the x-axis labels.

    Returns:
        None
    """
    # Count the occurrences of each sentiment and sort by sentiment index
    sentiment_counts = data['Sentiment'].value_counts().sort_index()  # Ensure sorted by index (0: Positive, 1: Neutral, 2: Negative)

    # Create a bar plot for sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_names, y=sentiment_counts.values, palette=["#3CB371", "#4682B4", "#FF4500"])  # Custom color palette
    plt.title("Sentiment Distribution", fontsize=16)
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def plot_training_metrics(history):
    """
    Visualize training and validation loss/accuracy over epochs.

    Args:
        history (dict): Dictionary containing:
            - 'train_loss' (list): Training loss values for each epoch.
            - 'val_loss' (list): Validation loss values for each epoch.
            - 'train_acc' (list): Training accuracy values for each epoch.
            - 'val_acc' (list): Validation accuracy values for each epoch.

    Returns:
        None
    """
    epochs = range(1, len(history['train_loss']) + 1)  # Create a range object for the number of epochs

    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)  # Create subplot 1 (1 row, 2 columns, first plot)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')  # Plot training loss
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')  # Plot validation loss
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)  # Create subplot 2 (1 row, 2 columns, second plot)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')  # Plot training accuracy
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o')  # Plot validation accuracy
    plt.title("Training and Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
