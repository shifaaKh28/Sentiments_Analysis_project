# text_preprocessing.py
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses the input text:
    - Removes URLs
    - Retains capital letters, special characters, and punctuation
    - Removes extra whitespace
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords (optional, depending on use case)
    tokens = text.split()  # Tokenize the text
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(tokens)
