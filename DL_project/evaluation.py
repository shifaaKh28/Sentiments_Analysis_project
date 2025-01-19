import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def eval_model(model, data_loader, loss_fn, device, n_examples, class_names):
    """
    Evaluates a model's performance on a dataset, including per-label statistics.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of data for evaluation.
        loss_fn (torch.nn.Module): Loss function to compute the error.
        device (torch.device): Device (CPU or GPU) where the computations will be performed.
        n_examples (int): Total number of examples in the dataset (for accuracy calculation).
        class_names (list): List of class names corresponding to the labels.

    Returns:
        dict: A dictionary containing evaluation metrics:
            - "accuracy" (torch.Tensor): Accuracy of the model on the dataset.
            - "loss" (float): Average loss on the dataset.
            - "precision" (float): Weighted precision score.
            - "recall" (float): Weighted recall score.
            - "f1_score" (float): Weighted F1 score.
            - "per_label_metrics" (dict): Precision, recall, and F1 score for each label.
    """
    model = model.eval()  # Set the model to evaluation mode
    losses = []  # List to store batch losses
    correct_predictions = 0  # Counter for correct predictions
    all_preds = []  # List to store all predictions
    all_targets = []  # List to store all targets

    # Create a progress bar for the data loader
    data_loader = tqdm(data_loader, desc="Evaluating", unit="batch")

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for d in data_loader:
            # Extract inputs and targets from the batch
            input_ids = d.get("input_ids")
            attention_mask = d.get("attention_mask")
            targets = d["targets"].to(device)

            # Forward pass depending on the model type
            if input_ids is not None and attention_mask is not None:  # For BERT-based models
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:  # For simpler models like SimpleNN
                inputs = d["features"].to(device)
                outputs = model(inputs)

            # Get the predicted class
            _, preds = torch.max(outputs, dim=1)

            # Compute loss
            loss = loss_fn(outputs, targets)

            # Update metrics
            correct_predictions += torch.sum(preds == targets)  # Count correct predictions
            losses.append(loss.item())  # Append batch loss

            # Store predictions and targets for additional metrics
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar with the current average loss
            data_loader.set_postfix(loss=np.mean(losses))

    # Calculate additional metrics
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Calculate per-label metrics
    per_label_precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    per_label_recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    per_label_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # Organize per-label metrics in a dictionary
    per_label_metrics = {
        class_names[i]: {
            "precision": per_label_precision[i],
            "recall": per_label_recall[i],
            "f1_score": per_label_f1[i],
        }
        for i in range(len(class_names))
    }

    return {
        "accuracy": correct_predictions.double() / n_examples,
        "loss": np.mean(losses),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_label_metrics": per_label_metrics
    }
