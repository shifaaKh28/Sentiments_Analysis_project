import os  # For checking if the model file exists
import pandas as pd
from text_preprocessing import preprocess_text
from data_split import split_dataset, save_splits
from sklearn.feature_extraction.text import TfidfVectorizer
from train_nn import SentimentClassifier, prepare_pytorch_data, train_nn_model, validate_model, save_model, test_model
import torch
import pickle

# Step 1: Load the dataset
df = pd.read_csv('data.csv')

# Step 2: Preprocess the text
df['Clean_Text'] = df['Text'].apply(preprocess_text)

# Simplify sentiment labels
sentiment_mapping = {
    'Positive': 'Positive', 'Joy': 'Positive', 'Excitement': 'Positive',
    'Happiness': 'Positive', 'Gratitude': 'Positive', 'Love': 'Positive',
    'Negative': 'Negative', 'Sadness': 'Negative', 'Anger': 'Negative',
    'Frustration': 'Negative', 'Fear': 'Negative', 'Disgust': 'Negative',
    'Neutral': 'Neutral', 'Calm': 'Neutral', 'Indifference': 'Neutral'
}

df['Sentiment'] = df['Sentiment'].str.strip().str.capitalize()
df['Simplified_Sentiment'] = df['Sentiment'].map(sentiment_mapping)
df = df.dropna(subset=['Simplified_Sentiment'])

# Step 3: Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

# Step 4: Extract Features with TF-IDF
def extract_features_tfidf(X_train, X_val, X_test):
    if os.path.exists("tfidf_vectorizer.pkl"):
        with open("tfidf_vectorizer.pkl", "rb") as file:
            tfidf_vectorizer = pickle.load(file)
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        with open("tfidf_vectorizer.pkl", "wb") as file:
            pickle.dump(tfidf_vectorizer, file)

    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer = extract_features_tfidf(
    X_train, X_val, X_test
)

# Prepare PyTorch Tensors
X_train_tensor, y_train_tensor = prepare_pytorch_data(X_train_tfidf, y_train)
X_val_tensor, y_val_tensor = prepare_pytorch_data(X_val_tfidf, y_val)
X_test_tensor, y_test_tensor = prepare_pytorch_data(X_test_tfidf, y_test)

# Step 5: Train or Load the Model
if os.path.exists("sentiment_nn_model.pth"):
    print("\nLoading existing model...")
    input_dim = X_train_tfidf.shape[1]
    hidden_dim = 128
    output_dim = 3
    trained_nn_model = SentimentClassifier(input_dim, hidden_dim, output_dim)
    trained_nn_model.load_state_dict(torch.load("sentiment_nn_model.pth"))
    trained_nn_model.eval()
    print("Model loaded successfully.")
else:
    print("\nTraining the model...")
    input_dim = X_train_tfidf.shape[1]
    hidden_dim = 128
    output_dim = 3
    trained_nn_model = train_nn_model(
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        input_dim, hidden_dim, output_dim
    )
    save_model(trained_nn_model, "sentiment_nn_model.pth")

# Validate the Model
validate_model(trained_nn_model, X_val_tensor, y_val_tensor)

# Test the Model
test_model(trained_nn_model, X_test_tensor, y_test_tensor)

# Test with a Custom Sentence
def preprocess_and_predict(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    tfidf_features = vectorizer.transform([cleaned_text])
    tfidf_tensor = torch.tensor(tfidf_features.toarray(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(tfidf_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()
    class_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return class_labels[predicted_class]


new_sentence = "I am happy."
predicted_sentiment = preprocess_and_predict(new_sentence, trained_nn_model, tfidf_vectorizer)
print(f"\nCustom Sentence: {new_sentence}")
print(f"Predicted Sentiment: {predicted_sentiment}")
