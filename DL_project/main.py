import os  # Import for checking file existence
from preprocessing import load_dataset, preprocess_dataset, tokenizer
from visualization import plot_sentiment_distribution, visualize_text_length, plot_training_metrics
from data_loader import create_data_loader
from models import SentimentClassifier
from training import train_epoch
from evaluation import eval_model
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn

def test_model(model, test_data_loader, loss_fn, device, n_examples):
    """
    Evaluate the model on the test dataset.
    """
    metrics = eval_model(model, test_data_loader, loss_fn, device, n_examples)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
    return metrics

def predict_sentiment(model, tokenizer, text, device, class_names):
    """
    Predict the sentiment of a given text using the trained model.
    """
    # Encode the input text
    encoding = tokenizer.encode_plus(
        text,
        max_length=50,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted_class = torch.max(outputs, dim=1)

    return class_names[predicted_class.item()]

def run_training(model, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, device, df_train, df_val, epochs):
    """
    Training and validation loop.
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_metrics = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
        val_acc, val_loss = val_metrics['accuracy'], val_metrics['loss']
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())

    return history

def main():
    # Load and preprocess the dataset
    df = load_dataset("data.csv")
    df, class_weights = preprocess_dataset(df)

    # Define class names for visualization
    class_names = ["Positive", "Neutral", "Negative"]

    # Visualization
    plot_sentiment_distribution(df, class_names)
    visualize_text_length(df, title="Filtered Dataset")

    # Data Split
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.30, shuffle=True, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=0.50, shuffle=True, random_state=42)

    # Data Loaders
    MAX_LEN = 50
    BATCH_SIZE = 16
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Model Initialization
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentClassifier(n_classes=3, pre_trained_model_name=PRE_TRAINED_MODEL_NAME).to(device)

    # Define Loss Function
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

    # Training setup
    model_path = "sentiment_classifier.pth"
    EPOCHS = 20
    LEARNING_RATE = 1e-5
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=4,
        num_training_steps=len(train_data_loader) * EPOCHS
    )

    # Check if model already trained
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading the model...")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    else:
        print(f"No trained model found at {model_path}. Training a new model...")
        history = run_training(model, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, device, df_train, df_val, EPOCHS)

        # Save training history
        plot_training_metrics(history)

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Evaluate on the test dataset
    print("\nEvaluating on Test Dataset:")
    test_metrics = test_model(model, test_data_loader, loss_fn, device, len(df_test))

    # Predict sentiments for sample texts
    sample_texts = [
        "I am so happy and excited about this project!",
        "This is a terrible day. I feel so sad.",
        "I'm not sure how I feel about this.",
        "Life is beautiful and full of joy."
    ]

    print("\nPredictions for Sample Texts:")
    for text in sample_texts:
        predicted_label = predict_sentiment(model, tokenizer, text, device, class_names)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {predicted_label}")

if __name__ == "__main__":
    main()
