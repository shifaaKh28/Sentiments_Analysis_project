import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# Visualization: Sentiment Distribution
def plot_sentiment_distribution(data, class_names):
    """
    Visualize the distribution of sentiments in the dataset.

    Args:
        data (DataFrame): The preprocessed dataset.
        class_names (list): List of sentiment class names for the x-axis labels.
    """
    plt.figure(figsize=(20, 6))
    custom_palette = ['#FF2400', 'teal', '#A52A2A', 'Seagreen', 'Dodgerblue', 'Purple', 'Gold', 'MediumVioletRed']

    sns.countplot(x='Sentiment', data=data, palette=custom_palette)
    plt.gca().set_xticklabels(class_names, rotation=45, ha="right")  # Add rotation for better readability
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()


# Visualization: Text Length Distribution
def visualize_text_length(data, title):
    """
    Visualize the distribution of text lengths in the dataset.

    Args:
        data (DataFrame): The preprocessed dataset.
        title (str): Title for the plot.
    """
    data['text_length'] = data['Text'].apply(len)

    plt.figure(figsize=(8, 4))
    custom_font = FontProperties(family='serif', style='normal', size=14, weight='bold')

    plt.hist(
        data['text_length'], bins=40, color='lightcoral', edgecolor='black', alpha=0.7, label='Text Length'
    )
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel("Text Length", fontsize=10, fontproperties=custom_font, color='black')
    plt.ylabel("Frequency", fontsize=10, fontproperties=custom_font, color='black')
    plt.title(f'Text Length Distribution for {title}', fontsize=12, fontproperties=custom_font, color='black')
    plt.show()



