import warnings

warnings.filterwarnings("ignore")

# Setup & Config
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Visualization settings
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
plt.rcParams["figure.figsize"] = (12, 8)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables
DATA_PATH = "data.csv"  # Update path if needed

# Load Dataset
def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset Preview:")
    print(df.head())
    return df

# Define the mapping dictionary
sentiment_to_stage = {
    # Positive sentiments
    "Positive": "Positive",
    "Excitement": "Positive",
    "Contentment": "Positive",
    "Joy": "Positive",
    "Happy": "Positive",
    "Happiness": "Positive",
    "Gratitude": "Positive",
    "Admiration": "Positive",
    "Love": "Positive",
    "Affection": "Positive",
    "Hopeful": "Positive",
    "Anticipation": "Positive",
    "Enthusiasm": "Positive",
    "Elated": "Positive",
    "Euphoria": "Positive",
    "Inspired": "Positive",
    "Optimism": "Positive",
    "Success": "Positive",
    "Accomplishment": "Positive",
    "Pride": "Positive",
    "Compassionate": "Positive",
    "Free-spirited": "Positive",
    "Playful": "Positive",
    "Thrill": "Positive",
    "Empowerment": "Positive",
    "Fulfillment": "Positive",
    "Wonder": "Positive",
    "Rejuvenation": "Positive",
    "Awe": "Positive",
    "Adventure": "Positive",
    "Harmony": "Positive",

    # Negative sentiments
    "Negative": "Negative",
    "Sad": "Negative",
    "Anger": "Negative",
    "Loneliness": "Negative",
    "Grief": "Negative",
    "Hate": "Negative",
    "Bitterness": "Negative",
    "Despair": "Negative",
    "Betrayal": "Negative",
    "Frustration": "Negative",
    "Isolation": "Negative",
    "Regret": "Negative",
    "Helplessness": "Negative",
    "Resentment": "Negative",
    "Disgust": "Negative",
    "Heartbreak": "Negative",
    "Jealousy": "Negative",
    "Sorrow": "Negative",
    "Loss": "Negative",
    "Envy": "Negative",
    "Melancholy": "Negative",
    "Anxiety": "Negative",
    "Darkness": "Negative",
    "Apprehension": "Negative",
    "Dismissive": "Negative",
    "Boredom": "Negative",

    # Neutral sentiments
    "Neutral": "Neutral",
    "Ambivalence": "Neutral",
    "Curiosity": "Neutral",
    "Indifference": "Neutral",
    "Acceptance": "Neutral",
    "Nostalgia": "Neutral",
    "Calmness": "Neutral",
    "Reflection": "Neutral",
    "Confusion": "Neutral",
    "Serenity": "Neutral",
    "Determination": "Neutral",
    "Mindfulness": "Neutral",
    "Pensive": "Neutral",
    "Compassion": "Neutral",
    "Contemplation": "Neutral",
    "Sympathy": "Neutral",
    "Motivation": "Neutral",
}

# Function to map sentiments to broader categories
def map_sentiment_to_stage(df, mapping):
    """
    Maps fine-grained sentiment labels to broader categories (Negative, Positive, Neutral).

    Args:
        df (pd.DataFrame): The input dataset with a 'Sentiment' column.
        mapping (dict): A dictionary mapping fine-grained sentiments to broader stages.

    Returns:
        pd.DataFrame: The updated dataset with a new 'Stage' column.
    """
    df['Stage'] = df['Sentiment'].replace(mapping)
    return df

# Preprocess Dataset
def preprocess_dataset(df):
    """
    Preprocesses the dataset:
    - Maps sentiments to broader stages
    - Converts stages to integer labels
    """
    # Map fine-grained sentiments to broader categories
    df = map_sentiment_to_stage(df, sentiment_to_stage)

    # Drop rows where the 'Stage' column is NaN (unmapped sentiments)
    df = df.dropna(subset=["Stage"])

    # Ensure 'Stage' is a category for efficient storage
    df["Stage"] = df["Stage"].astype("category")

    # Create a mapping for Stage -> Integer
    stage_to_int = {stage: idx for idx, stage in enumerate(df["Stage"].cat.categories)}
    print("Stage to Integer Mapping:", stage_to_int)

    # Map Stage column to integers
    df["Stage"] = df["Stage"].replace(stage_to_int)

    # Print unique stages and their counts
    print("Unique Stages:", df["Stage"].unique())
    print("Stage Counts:")
    print(df["Stage"].value_counts())

    return df, stage_to_int

# Tokenizer Configuration
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def tokenize_sample(sample_text):
    """
    Tokenize a single text sample and return its tokens and token IDs.
    """
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids
