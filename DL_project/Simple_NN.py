import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Load pre-split datasets
def load_splits():
    df_train = pd.read_csv("train_split.csv", encoding='latin1')
    df_test = pd.read_csv("test_split.csv", encoding='latin1')
    return df_train, df_test


# Improved Neural Network Architecture
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Predict sentiment for new examples
def predict_examples(model, vectorizer, examples, device):
    model.eval()
    print("\nPredicting Sentiments for Examples:")
    for text in examples:
        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])
        text_tensor = torch.tensor(text_vectorized.toarray(), dtype=torch.float32).to(device)

        # Predict sentiment
        with torch.no_grad():
            output = model(text_tensor)
            _, predicted = torch.max(output, 1)

        # Map predicted class to sentiment
        sentiment_mapping = {0: "Positive", 1: "Neutral", 2: "Negative"}
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment_mapping[predicted.item()]}")
        print("-" * 50)


# Train and evaluate the improved NN
def simple_nn_evaluation():
    # Load pre-split data
    df_train, df_test = load_splits()

    # Split validation set from training data
    df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=42)

    # Separate features and labels
    X_train, y_train = df_train['Text'], df_train['Sentiment']
    X_val, y_val = df_val['Text'], df_val['Sentiment']
    X_test, y_test = df_test['Text'], df_test['Sentiment']

    # TF-IDF Vectorizer
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Convert TF-IDF matrices to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model
    input_dim = X_train_tfidf.shape[1]
    num_classes = len(y_train.unique())
    model = ImprovedNN(input_dim=input_dim, num_classes=num_classes)
    model = model.to("cpu")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Training loop with Early Stopping
    print("Training the Neural Network...")
    num_epochs = 10
    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation on Train, Validation, and Test sets
    def evaluate(loader, dataset_name):
        model.eval()
        total = 0
        correct = 0
        total_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

                y_true.extend(y_batch.tolist())
                y_pred.extend(predicted.tolist())

        accuracy = correct / total
        error = 1 - accuracy
        print(f"\n{dataset_name} Results:")
        print(classification_report(y_true, y_pred, target_names=["Positive", "Neutral", "Negative"]))
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        print(f"{dataset_name} Error: {error:.4f}")
        return accuracy, error

    train_accuracy, train_error = evaluate(train_loader, "Train")
    val_accuracy, val_error = evaluate(val_loader, "Validation")
    test_accuracy, test_error = evaluate(test_loader, "Test")

    # Summary of results
    print("\nSummary:")
    print(f"Train Accuracy: {train_accuracy:.4f}, Error: {train_error:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}, Error: {val_error:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Error: {test_error:.4f}")

    # Example predictions
    examples = [
        "I love this new feature in the app!",
        "The weather today is very depressing.",
        "I'm not sure how to feel about this update.",
        "The service was fantastic and exceeded expectations!",
        "This is just another average day."
    ]
    predict_examples(model, vectorizer, examples, device="cpu")

# Main function
if __name__ == "__main__":
    simple_nn_evaluation()
