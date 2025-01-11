import torch #Provides tensor operations and deep learning functionalities.
from torch import nn #Contains neural network layers and components.
from transformers import AutoModel #Loads pre-trained transformer models.

class SentimentClassifier(nn.Module):
    """
    Sentiment classification model using a pre-trained transformer (e.g., BERT).

    Attributes:
        bert (transformers.AutoModel): Pre-trained transformer model for feature extraction.
        drop (nn.Dropout): Dropout layer to reduce overfitting.
        out (nn.Linear): Fully connected layer for classification.
    """
    def __init__(self, n_classes, pre_trained_model_name):
        """
        Initializes the sentiment classifier.

        Args:
            n_classes (int): Number of output classes.
            pre_trained_model_name (str): Name of the pre-trained transformer model to load.
        """
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pre_trained_model_name)  # Load pre-trained transformer
        self.drop = nn.Dropout(p=0.5)  # Add dropout for regularization
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # Output layer

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens are attention-worthy.

        Returns:
            torch.Tensor: Logits for each class.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # Use the representation of the [CLS] token for classification
        logits = outputs[0][:, 0, :]  # First element is the sequence output
        output = self.drop(logits)
        return self.out(output)


class SimpleNN(nn.Module):
    """
    A simple fully connected neural network for sentiment classification.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): Activation function.
        fc2 (nn.Linear): Output layer for classification.
    """
    def __init__(self, input_size, hidden_size, n_classes):
        """
        Initializes the simple neural network.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of units in the hidden layer.
            n_classes (int): Number of output classes.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, n_classes)  # Hidden to output layer

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# Usage:
# - Use `SentimentClassifier` for the advanced model with BERT.
# - Use `SimpleNN` for a basic fully connected neural network.
