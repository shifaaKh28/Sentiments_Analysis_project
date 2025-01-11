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
    # Comprehensive sentiment mapping
    sentiment_mapping = {
        # Positive
        "Positive": "Positive", "Excitement": "Positive", "Contentment": "Positive", "Joy": "Positive",
        "Happy": "Positive", "Hopeful": "Positive", "Gratitude": "Positive", "Admiration": "Positive",
        "Affection": "Positive", "Serenity": "Positive", "Elation": "Positive", "Enthusiasm": "Positive",
        "Euphoria": "Positive", "Amusement": "Positive", "Kind": "Positive", "Pride": "Positive",
        "Tenderness": "Positive", "Arousal": "Positive", "Fulfillment": "Positive", "Reverence": "Positive",
        "Overjoyed": "Positive", "Motivation": "Positive", "JoyfulReunion": "Positive", "Satisfaction": "Positive",
        "Blessed": "Positive", "Appreciation": "Positive", "Confidence": "Positive",
        "Accomplishment": "Positive", "Wonderment": "Positive", "Optimism": "Positive", "Enchantment": "Positive",
        "Intrigue": "Positive", "PlayfulJoy": "Positive", "Mindfulness": "Positive", "Elegance": "Positive",
        "Whimsy": "Positive", "Pensive": "Positive", "Harmony": "Positive", "Creativity": "Positive",
        "Radiance": "Positive", "Wonder": "Positive", "Rejuvenation": "Positive", "Coziness": "Positive",
        "Adventure": "Positive", "Awe": "Positive", "FestiveJoy": "Positive", "InnerJourney": "Positive",
        "Freedom": "Positive", "Dazzle": "Positive", "ArtisticBurst": "Positive", "CulinaryOdyssey": "Positive",
        "Resilience": "Positive", "Spark": "Positive", "Marvel": "Positive", "Positivity": "Positive",
        "Kindness": "Positive", "Friendship": "Positive", "Success": "Positive", "Exploration": "Positive",
        "Amazement": "Positive", "Romance": "Positive", "Captivation": "Positive", "Tranquility": "Positive",
        "Grandeur": "Positive", "Emotion": "Positive", "Energy": "Positive", "Celebration": "Positive",
        "Charm": "Positive", "Ecstasy": "Positive", "Colorful": "Positive", "Hypnotic": "Positive",
        "Connection": "Positive", "Iconic": "Positive", "Journey": "Positive", "Engagement": "Positive",
        "Touched": "Positive", "Triumph": "Positive", "Heartwarming": "Positive", "Solace": "Positive",
        "Breakthrough": "Positive", "Vibrancy": "Positive", "Mesmerizing": "Positive", "Culinary Adventure": "Positive",
        "Winter Magic": "Positive", "Thrilling Journey": "Positive", "Nature's Beauty": "Positive",
        "Celestial Wonder": "Positive", "Creative Inspiration": "Positive", "Runway Creativity": "Positive",
        "Ocean's Freedom": "Positive", "Whispers of the Past": "Positive", "Relief": "Positive",
        "Mischievous": "Positive", "Inspired": "Positive", "Zest": "Positive", "Proud": "Positive",
        "Grateful": "Positive", "Empathetic": "Positive", "Compassionate": "Positive", "Playful": "Positive",
        "Free-spirited": "Positive", "Confident": "Positive",

        # Negative
        "Negative": "Negative", "Sadness": "Negative", "Loneliness": "Negative", "Grief": "Negative",
        "Anger": "Negative", "Fear": "Negative", "Frustration": "Negative", "Bitterness": "Negative",
        "Regret": "Negative", "Despair": "Negative", "Hate": "Negative", "Betrayal": "Negative",
        "Suffering": "Negative", "EmotionalStorm": "Negative", "Isolation": "Negative", "Disappointment": "Negative",
        "LostLove": "Negative", "Exhaustion": "Negative", "Sorrow": "Negative", "Darkness": "Negative",
        "Desperation": "Negative", "Ruins": "Negative", "Desolation": "Negative", "Loss": "Negative",
        "Heartache": "Negative", "Jealousy": "Negative", "Resentment": "Negative", "Boredom": "Negative",
        "Anxiety": "Negative", "Intimidation": "Negative", "Helplessness": "Negative", "Envy": "Negative",
        "Fearful": "Negative", "Apprehensive": "Negative", "Overwhelmed": "Negative", "Devastated": "Negative",
        "Envious": "Negative", "Dismissive": "Negative", "Bitter": "Negative","Melancholy": "Negative","Pressure": "Negative",


        # Neutral
        "Neutral": "Neutral", "Acceptance": "Neutral", "Adoration": "Neutral",
        "Anticipation": "Neutral", "Calmness": "Neutral", "Confusion": "Neutral",
        "Indifference": "Neutral", "Curiosity": "Neutral", "Numbness": "Neutral",
        "Nostalgia": "Neutral", "Ambivalence": "Neutral", "Determination": "Neutral", "Reflection": "Neutral",
        "Obstacle": "Neutral", "Sympathy": "Neutral", "Renewed Effort": "Neutral",
        "Miscalculation": "Neutral", "Challenge": "Neutral",
        "Envisioning History": "Neutral", "Imagination": "Neutral"
    }

 # Debugging: Print unique sentiments before mapping
    print("Unique sentiments in dataset before mapping:", df["Sentiment"].unique())

    # Strip whitespace from sentiment values
    df["Sentiment"] = df["Sentiment"].str.strip()

    # Apply sentiment mapping
    df["Sentiment"] = df["Sentiment"].map(sentiment_mapping)

    # Debugging: Log unmapped values
    unmapped_count = df["Sentiment"].isna().sum()
    print(f"Number of unmapped sentiments: {unmapped_count}")

    # Drop rows with unmapped sentiments (NaN)
    df = df.dropna(subset=["Sentiment"])

    # Map class names to integers
    class_to_int = {"Positive": 0, "Neutral": 1, "Negative": 2}
    df["Sentiment"] = df["Sentiment"].map(class_to_int)

    # Ensure the Sentiment column is of type integer
    df["Sentiment"] = df["Sentiment"].astype(int)

    # Calculate class weights for imbalanced dataset
    unique_classes = np.unique(df["Sentiment"])
    print("Unique classes in Sentiment after mapping:", unique_classes)
    print("Sentiment counts:", df["Sentiment"].value_counts())

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=df["Sentiment"].values
    )

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
