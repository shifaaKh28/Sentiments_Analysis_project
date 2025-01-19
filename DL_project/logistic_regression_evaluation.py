import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")


# Load pre-split datasets
def load_splits():
    df_train = pd.read_csv("train_split.csv", encoding='latin1')
    df_val = pd.read_csv("val_split.csv", encoding='latin1')
    df_test = pd.read_csv("test_split.csv", encoding='latin1')
    return df_train, df_val, df_test


# Train and evaluate Logistic Regression model
def logistic_regression_evaluation():
    # Load pre-split data
    df_train, df_val, df_test = load_splits()

    # Separate features and labels
    X_train, y_train = df_train['Text'], df_train['Sentiment']
    X_val, y_val = df_val['Text'], df_val['Sentiment']
    X_test, y_test = df_test['Text'], df_test['Sentiment']

    # TF-IDF Vectorizer
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')

    # Fit TF-IDF on training data and transform all splits
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Debug: Check the dimensions of the vectorized data
    print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
    print(f"Shape of X_val_tfidf: {X_val_tfidf.shape}")
    print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")

    # Logistic Regression Model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)

    results = {}

    # Function to evaluate a dataset
    def evaluate(dataset_name, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        error = 1 - accuracy

        print(f"\n{dataset_name} Results:")
        print(classification_report(y, y_pred, target_names=["Positive", "Neutral", "Negative"]))
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        print(f"{dataset_name} Error: {error:.4f}")

        results[dataset_name] = {
            "accuracy": accuracy,
            "error": error
        }

    # Evaluate train, validation, and test datasets
    evaluate("Train", X_train_tfidf, y_train)
    evaluate("Validation", X_val_tfidf, y_val)
    evaluate("Test", X_test_tfidf, y_test)

    return model, vectorizer, results


# Predict sentiment for example sentences
def predict_examples(model, vectorizer):
    # Example sentences
    example_sentences = [
        "I love studying at Ariel University! The professors are amazing.",
        "This project is so difficult; I can't figure it out.",
        "I'm neutral about this subject, it's neither good nor bad.",
        "What a great experience learning about deep learning!",
        "The campus cafeteria food could be better.",
        "I hate FLAFEL"
    ]

    # Vectorize example sentences
    example_tfidf = vectorizer.transform(example_sentences)

    # Predict sentiments
    predictions = model.predict(example_tfidf)

    # Map predictions to sentiment labels
    sentiment_labels = {0: "Positive", 1: "Neutral", 2: "Negative"}
    for sentence, prediction in zip(example_sentences, predictions):
        print(f"Sentence: \"{sentence}\"")
        print(f"Predicted Sentiment: {sentiment_labels[prediction]}")
        print()


# Main function
if __name__ == "__main__":
    # Train and evaluate the logistic regression model
    model, vectorizer, results = logistic_regression_evaluation()

    # Print summary
    print("\nSummary:")
    for dataset, metrics in results.items():
        print(f"{dataset} Accuracy: {metrics['accuracy']:.4f}, Error: {metrics['error']:.4f}")

    # Test example sentences
    print("\nTesting Example Sentences:")
    predict_examples(model, vectorizer)
