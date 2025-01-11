import torch
from tqdm import tqdm
import numpy as np #For numerical computations.

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of data.
        loss_fn (torch.nn.Module): Loss function to compute the error.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        device (torch.device): Device (CPU or GPU) where the computations will be performed.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler for adaptive learning rates.
        n_examples (int): Total number of examples in the dataset (for accuracy calculation).

    Returns:
        tuple: (accuracy, average_loss)
            - accuracy (torch.Tensor): Training accuracy for the epoch.
            - average_loss (float): Average training loss for the epoch.
    """
    model = model.train()  # Set the model to training mode
    losses = []  # List to store batch losses
    correct_predictions = 0  # Counter for correct predictions

    # Create a progress bar for the data loader
    data_loader = tqdm(data_loader, desc="Training", unit="batch")

    for d in data_loader:
        # Extract inputs and targets from the batch
        input_ids = d.get("input_ids")
        attention_mask = d.get("attention_mask")
        targets = d["targets"].squeeze().to(device)  # Move targets to the specified device

        # Forward pass depending on the model type
        if input_ids is not None and attention_mask is not None:  # For models like BERT
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
        correct_predictions += torch.sum(preds == targets)  # Count correct predictions
        losses.append(loss.item())  # Append batch loss

        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        if scheduler:  # Update the learning rate if scheduler is provided
            scheduler.step()
        optimizer.zero_grad()  # Reset gradients for the next iteration

        # Update progress bar with the current average loss
        data_loader.set_postfix(loss=np.mean(losses))

    # Calculate overall accuracy and average loss
    return correct_predictions.double() / n_examples, np.mean(losses)
