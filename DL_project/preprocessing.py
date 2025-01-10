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
from transformers import DistilBertTokenizer  # Updated to DistilBertTokenizer
from sklearn.utils import class_weight
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
DATA_PATH = "sentimentdataset.csv"  # Update path if needed

# Load Dataset
def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset Preview:")
    print(df.head())
    return df

# Preprocess Dataset
def preprocess_dataset(df):
    # Map detailed sentiments into three categories: Positive, Neutral, Negative
    sentiment_mapping = {
        " Positive  ": "Positive", " Excitement ": "Positive", " Contentment ": "Positive",
        " Joy ": "Positive", " Happy ": "Positive", " Hopeful ": "Positive", " Gratitude ": "Positive",
        " Admiration ": "Positive", " Affection ": "Positive", " Serenity ": "Positive", " Elation ": "Positive",
        " Enthusiasm ": "Positive", " Euphoria ": "Positive", " Amusement ": "Positive",
        " Sad ": "Negative", " Loneliness ": "Negative", " Grief ": "Negative", " Anger ": "Negative",
        " Fear ": "Negative", " Frustration ": "Negative", " Bitterness ": "Negative",
        " Regret ": "Negative", " Despair ": "Negative", " Hate ": "Negative", " Betrayal ": "Negative",
        " Neutral ": "Neutral", " Confusion ": "Neutral", " Indifference ": "Neutral",
        " Curiosity ": "Neutral", " Acceptance ": "Neutral", " Calmness ": "Neutral"
    }

    # Apply sentiment mapping to the dataset
    df["Sentiment"] = df["Sentiment"].map(sentiment_mapping)

    # Drop rows with unmapped sentiments (if any)
    df = df.dropna(subset=["Sentiment"])

    # Map class names to integers
    class_to_int = {
        "Positive": 0,
        "Neutral": 1,
        "Negative": 2
    }

    # Apply the mapping to the 'Sentiment' column
    df["Sentiment"] = df["Sentiment"].map(class_to_int)

    # Calculate class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["Sentiment"]),
        y=df["Sentiment"].values
    )

    # Print unique sentiment labels and their counts
    print("Unique Sentiments:", df["Sentiment"].unique())
    print("Sentiment Counts:")
    print(df["Sentiment"].value_counts())

    return df, class_weights

# Tokenizer Configuration
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)  # Updated tokenizer

def tokenize_sample(sample_text):
    """
    Tokenize a single text sample and return its tokens and token IDs.
    """
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids
