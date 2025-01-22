import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Visualization: Sentiment Distribution
# Visualization: Sentiment Distribution
def plot_sentiment_distribution(data, class_names):
    """
    Visualize the distribution of sentiments in the dataset.

    Args:
        data (DataFrame): The preprocessed dataset.
        class_names (list): List of sentiment class names for the x-axis labels.
    """
    sentiment_counts = data['Sentiment'].value_counts().sort_index()  # Ensure sorted by index (0: Positive, 1: Neutral, 2: Negative)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_names, y=sentiment_counts.values, palette=["#3CB371", "#4682B4", "#FF4500"])
    plt.title("Sentiment Distribution", fontsize=16)
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Visualization: Training and Validation Metrics
def plot_training_metrics(history):
    """
    Visualize training and validation loss/accuracy over epochs.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', and 'val_acc'.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
