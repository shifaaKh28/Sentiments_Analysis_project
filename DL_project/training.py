import torch
from tqdm import tqdm
import numpy as np

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    data_loader = tqdm(data_loader, desc="Training", unit="batch")
    for d in data_loader:
        input_ids = d.get("input_ids")
        attention_mask = d.get("attention_mask")
        targets = d["targets"].squeeze().to(device)

        if input_ids is not None and attention_mask is not None:  # For models like BERT
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:  # For SimpleNN or similar models
            inputs = d["features"].to(device)
            outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        data_loader.set_postfix(loss=np.mean(losses))

    return correct_predictions.double() / n_examples, np.mean(losses)
