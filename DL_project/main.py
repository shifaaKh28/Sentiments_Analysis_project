# =============================
# Import Libraries
# =============================

# Data manipulation and analysis
import pandas as pd  # For handling and manipulating tabular data

# Data visualization
import matplotlib.pyplot as plt  # For creating static plots
import seaborn as sns  # For creating advanced statistical plots

# Text preprocessing
import nltk  # Natural Language Toolkit for text processing
import re  # Regular expressions for text cleaning
from nltk.corpus import stopwords  # Stopwords for filtering common words
from nltk.stem import WordNetLemmatizer  # Lemmatizer for reducing words to their base form

# Machine learning utilities
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation

# Miscellaneous
import numpy as np  # For numerical operations
import warnings  # To suppress unnecessary warnings in output
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

# Download NLTK resources (if not already downloaded)
nltk.download('stopwords')  # Download stopwords list
nltk.download('wordnet')  # Download WordNet for lemmatization
nltk.download('punkt')  # Download tokenizer for text tokenization

# =============================
# Load the Dataset
# =============================

# Load the dataset
df = pd.read_csv('data.csv')

# Drop rows with missing 'Text' values
df = df.dropna(subset=['Text'])

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Standardize sentiment labels (remove extra spaces and handle case inconsistencies)
df['Sentiment'] = df['Sentiment'].str.strip().str.capitalize()

# Define mapping for sentiment categories
sentiment_mapping = {
    # Positive emotions
    'Positive': 'Positive', 'Happiness': 'Positive', 'Joy': 'Positive',
    'Admiration': 'Positive', 'Affection': 'Positive', 'Excitement': 'Positive',
    'Contentment': 'Positive', 'Gratitude': 'Positive', 'Hope': 'Positive',
    'Elation': 'Positive', 'Confidence': 'Positive', 'Pride': 'Positive',
    'Inspired': 'Positive', 'Optimism': 'Positive', 'Ecstasy': 'Positive',
    'Celebration': 'Positive', 'Success': 'Positive', 'Heartwarming': 'Positive',

    # Negative emotions
    'Negative': 'Negative', 'Anger': 'Negative', 'Fear': 'Negative',
    'Sadness': 'Negative', 'Disgust': 'Negative', 'Frustration': 'Negative',
    'Jealousy': 'Negative', 'Boredom': 'Negative', 'Anxiety': 'Negative',
    'Grief': 'Negative', 'Loneliness': 'Negative', 'Heartbreak': 'Negative',
    'Suffering': 'Negative', 'Desperation': 'Negative', 'Regret': 'Negative',
    'Hate': 'Negative', 'Isolation': 'Negative', 'Loss': 'Negative',

    # Neutral emotions
    'Neutral': 'Neutral', 'Curiosity': 'Neutral', 'Calmness': 'Neutral',
    'Acceptance': 'Neutral', 'Ambivalence': 'Neutral', 'Reflection': 'Neutral',
    'Mindfulness': 'Neutral', 'Harmony': 'Neutral', 'Tranquility': 'Neutral',
    'Contemplation': 'Neutral', 'Resilience': 'Neutral', 'Indifference': 'Neutral'
}

# Map the detailed sentiments to broader categories
df['Simplified_Sentiment'] = df['Sentiment'].map(sentiment_mapping)

# Drop rows with unmapped sentiments
df = df.dropna(subset=['Simplified_Sentiment'])

# Display unique values in the simplified sentiment column
print("\nSimplified Sentiments after re-mapping:")
print(df['Simplified_Sentiment'].value_counts())

# Plot the distribution of simplified sentiments
sns.countplot(x='Simplified_Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Simplified Sentiment Distribution')
plt.show()

# =============================
# Text Preprocessing
# =============================

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()  # For reducing words to their base form
stop_words = set(stopwords.words('english'))  # Common stopwords in English

# Define a function to preprocess the text
def preprocess_text(text):
    """
    Preprocesses the input text:
    - Converts to lowercase
    - Removes URLs, special characters, and numbers
    - Tokenizes, removes stopwords, and lemmatizes
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""  # Return an empty string for invalid inputs
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(tokens)

# Apply the preprocessing function to the 'Text' column
df['Clean_Text'] = df['Text'].apply(preprocess_text)

# Display the first few rows of the cleaned data
print("\nFirst few rows after text preprocessing:")
print(df[['Text', 'Clean_Text']].head())

# =============================
# Splitting the Dataset
# =============================

# Features and target
X = df['Clean_Text']  # Preprocessed text data
y = df['Simplified_Sentiment']  # Target sentiment labels

# Map sentiment labels to numerical values
label_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
y = y.map(label_mapping)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print dataset split statistics
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =============================
# Feature Extraction with TF-IDF
# =============================

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Print the shapes of the resulting matrices
print(f"\nTF-IDF matrix shape (training): {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (testing): {X_test_tfidf.shape}")

# =============================
# Baseline Model
# =============================

# Predict the majority class (most frequent class in the training set)
majority_class = y_train.mode()[0]
y_pred_baseline = np.full(shape=y_test.shape, fill_value=majority_class)

# Evaluate the baseline model
print("\nBaseline Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=label_mapping.keys()))
