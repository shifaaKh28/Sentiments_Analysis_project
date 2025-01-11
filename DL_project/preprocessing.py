import warnings
warnings.filterwarnings("ignore")

# Setup & Config
import transformers
from transformers import DistilBertTokenizer  # Updated to DistilBertTokenizer
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Visualization settings
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
plt.rcParams["figure.figsize"] = (12, 8)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)  # Updated tokenizer

# Tokenizer utility function
def tokenize_sample(sample_text):
    """
    Tokenize a single text sample and return its tokens and token IDs.
    """
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids

# Load Dataset
def load_dataset(path):
    """
    Load dataset from the specified path and display its preview.
    """
    try:
        df = pd.read_csv(path)
        print("Dataset Preview:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# Preprocess Dataset
def preprocess_dataset(df):
    """
    Preprocess the dataset by mapping sentiments, handling unmapped values,
    and computing class weights.
    """
    sentiment_mapping = {
        # Positive sentiments
        "Positive": "Positive", "Excitement": "Positive", "Contentment": "Positive",
        "Joy": "Positive", "Happy": "Positive", "Hopeful": "Positive", "Gratitude": "Positive",
        "Admiration": "Positive", "Affection": "Positive", "Serenity": "Positive",
        "Pride": "Positive", "Euphoria": "Positive", "Amusement": "Positive",
        "Tenderness": "Positive", "Adventure": "Positive", "Freedom": "Positive",
        "Inspired": "Positive", "Grateful": "Positive",

        # Negative sentiments
        "Negative": "Negative", "Sadness": "Negative", "Loneliness": "Negative",
        "Grief": "Negative", "Anger": "Negative", "Fear": "Negative", "Frustration": "Negative",
        "Regret": "Negative", "Despair": "Negative", "Hate": "Negative", "Betrayal": "Negative",
        "Sorrow": "Negative", "Loss": "Negative", "Jealousy": "Negative",

        # Neutral sentiments
        "Neutral": "Neutral", "Acceptance": "Neutral", "Anticipation": "Neutral",
        "Calmness": "Neutral", "Confusion": "Neutral", "Curiosity": "Neutral",
        "Reflection": "Neutral", "Sympathy": "Neutral", "Ambivalence": "Neutral",
        "Indifference": "Neutral",
    }

    # Debugging: Display unique sentiments before mapping
    print("Unique sentiments in dataset before mapping:", df["Sentiment"].unique())

    # Strip whitespace from sentiment values
    df["Sentiment"] = df["Sentiment"].str.strip()

    # Apply sentiment mapping
    df["Sentiment"] = df["Sentiment"].map(sentiment_mapping)

    # Log unmapped values
    unmapped_count = df["Sentiment"].isna().sum()
    if unmapped_count > 0:
        print(f"Number of unmapped sentiments: {unmapped_count}")
        print("Unmapped rows:")
        print(df[df["Sentiment"].isna()])

    # Drop rows with unmapped sentiments (NaN)
    df = df.dropna(subset=["Sentiment"])

    # Map class names to integers
    class_to_int = {"Positive": 0, "Neutral": 1, "Negative": 2}
    df["Sentiment"] = df["Sentiment"].map(class_to_int)

    # Ensure the Sentiment column is of type integer
    df["Sentiment"] = df["Sentiment"].astype(int)

    # Debugging: Display unique classes after mapping
    unique_classes = df["Sentiment"].unique()
    print("Unique classes in Sentiment after mapping:", unique_classes)
    print("Sentiment counts:", df["Sentiment"].value_counts())

    # Calculate class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=df["Sentiment"].values
    )

    return df, class_weights
