import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []

    data_loader = tqdm(data_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for d in data_loader:
            input_ids = d.get("input_ids")
            attention_mask = d.get("attention_mask")
            targets = d["targets"].to(device)

            if input_ids is not None and attention_mask is not None:  # For BERT-based models
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

            # Store predictions and targets for additional metrics
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            data_loader.set_postfix(loss=np.mean(losses))

    # Calculate additional metrics
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {
        "accuracy": correct_predictions.double() / n_examples,
        "loss": np.mean(losses),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
