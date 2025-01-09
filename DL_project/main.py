import os  # Import for checking file existence
from preprocessing import load_dataset, preprocess_dataset, tokenizer
from visualization import plot_sentiment_distribution, visualize_text_length
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
    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        n_examples
    )
    print(f"Test Accuracy: {test_acc.item():.4f}")
    return test_acc.item()


def predict_sentiment(model, tokenizer, text, device, stage_names):
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
        truncation=True,  # Explicit truncation
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted_class = torch.max(outputs, dim=1)

    return stage_names[predicted_class.item()]


def main():
    # Load and preprocess the dataset
    df = load_dataset("data.csv")
    df = preprocess_dataset(df)

    # Define broad sentiment stages
    stage_names = ["Negative", "Neutral", "Positive"]

    # Visualization
    plot_sentiment_distribution(df, stage_names)
    visualize_text_length(df, title="Dataset")

    # Data Split
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.30, shuffle=True, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=0.50, shuffle=True, random_state=42)

    # Data Loaders
    MAX_LEN = 50
    BATCH_SIZE = 8
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Model Initialization
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentClassifier(n_classes=3, pre_trained_model_name=PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    # Define Loss Function (always available)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Check if the model is already trained
    model_path = "sentiment_classifier.pth"
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading the model...")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    else:
        print(f"No trained model found at {model_path}. Training a new model...")

        # Hyperparameters, Optimizer, and Scheduler
        EPOCHS = 20  # Adjust this to your needs
        LEARNING_RATE = 2e-5
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=total_steps
        )

        # Training and Validation Loop
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

            val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Evaluate on the test dataset
    print("\nEvaluating on Test Dataset:")
    test_acc = test_model(model, test_data_loader, loss_fn, device, len(df_test))

    # Predict sentiments for sample texts
    sample_texts = [
        "I am so happy and excited about this project!",
        "This is a terrible day. I feel so sad.",
        "I'm not sure how I feel about this.",
        "Life is beautiful and full of joy."
    ]

    print("\nPredictions for Sample Texts:")
    for text in sample_texts:
        predicted_label = predict_sentiment(model, tokenizer, text, device, stage_names)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {predicted_label}")


if __name__ == "__main__":
    main()
