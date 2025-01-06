import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# Neural Network Model Definition
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Prepare Data for PyTorch
def prepare_pytorch_data(X, y):
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    return X_tensor, y_tensor


# Train the Neural Network
def train_nn_model(X_train, y_train, X_val, y_val, input_dim, hidden_dim, output_dim, epochs=50, lr=0.001):
    model = SentimentClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_accuracy = accuracy_score(
                y_val.numpy(), torch.argmax(y_val_pred, dim=1).numpy()
            )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, "
                  f"Validation Loss: {val_loss.item():.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}")

    return model


# Validate the Model
def validate_model(model, X_val, y_val):
    with torch.no_grad():
        y_val_pred = model(X_val)
        y_val_pred_labels = torch.argmax(y_val_pred, dim=1)

        print("\nValidation Performance:")
        print(f"Accuracy: {accuracy_score(y_val.numpy(), y_val_pred_labels.numpy()):.4f}")
        print("Classification Report:")
        print(classification_report(y_val.numpy(), y_val_pred_labels.numpy(),
                                    target_names=['Negative', 'Neutral', 'Positive']))


# Test the Model on the Test Set
def test_model(model, X_test, y_test):
    """
    Evaluates the model on the test set.

    Args:
        model: The trained PyTorch model.
        X_test: TF-IDF features for the test set (PyTorch tensor).
        y_test: True labels for the test set (PyTorch tensor).

    Returns:
        None
    """
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred_labels = torch.argmax(y_test_pred, dim=1)

        print("\nTest Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test.numpy(), y_test_pred_labels.numpy()):.4f}")
        print("Classification Report:")
        print(classification_report(y_test.numpy(), y_test_pred_labels.numpy(),
                                    target_names=['Negative', 'Neutral', 'Positive']))


# Save the Trained Model
def save_model(model, model_path="sentiment_nn_model.pth"):
    torch.save(model.state_dict(), model_path)
    print(f"\nNeural Network model saved as '{model_path}'")