import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def baseline_model(df_train=None, df_val=None, df_test=None, example_sentences=None):
    """
    Baseline model that predicts all samples as the most frequent class.
    """
    results = {}

    # Determine the most frequent class
    if df_train is not None:
        most_frequent_class = df_train['Sentiment'].value_counts().idxmax()
    else:
        # Default to class '0' if training data is not provided
        most_frequent_class = 0

    print(f"Most Frequent Class: {most_frequent_class}\n")

    # Evaluation function for datasets
    def evaluate(dataset, dataset_name):
        y_true = dataset['Sentiment']
        y_pred = [most_frequent_class] * len(y_true)

        accuracy = accuracy_score(y_true, y_pred)
        error = 1 - accuracy  # Error is 1 - Accuracy

        print(f"\n{dataset_name} Results:")
        print(classification_report(y_true, y_pred, target_names=["Positive", "Neutral", "Negative"]))
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        print(f"{dataset_name} Error: {error:.4f}")

        results[dataset_name] = {
            "accuracy": accuracy,
            "error": error
        }

    # Evaluate datasets if provided
    if df_train is not None:
        evaluate(df_train, "Train")
    if df_val is not None:
        evaluate(df_val, "Validation")
    if df_test is not None:
        evaluate(df_test, "Test")

    # Example sentences prediction
    if example_sentences:
        print("\nExample Sentences Prediction:")
        for sentence in example_sentences:
            print(f"Sentence: {sentence}")
            print(f"Predicted Sentiment: {most_frequent_class} (Positive)\n")

    return results


if __name__ == "__main__":
    # Load datasets
    df_train = pd.read_csv("train_split.csv", encoding='latin1')
    df_val = pd.read_csv("val_split.csv", encoding='latin1')
    df_test = pd.read_csv("test_split.csv", encoding='latin1')

    print("\nRunning Baseline Model on Train, Validation, and Test Datasets:")
    results = baseline_model(df_train, df_val, df_test)

    # Print summary
    print("\nSummary:")
    for dataset, metrics in results.items():
        print(f"{dataset} Accuracy: {metrics['accuracy']:.4f}, Error: {metrics['error']:.4f}")

    # Example sentences
    example_sentences = [
        "I love the new updates to the app! It works flawlessly now.",
        "The service was really disappointing and slow.",
        "I'm not sure how I feel about the new feature."
    ]

    print("\nRunning Baseline Model on Test Dataset with Example Sentences:")
    baseline_model(df_train=None, df_val=None, df_test=df_test, example_sentences=example_sentences)
