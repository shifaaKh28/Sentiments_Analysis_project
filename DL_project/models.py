import torch
from torch import nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, pre_trained_model_name):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pre_trained_model_name)  # Pre-trained model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # Use the representation of the [CLS] token
        logits = outputs[0][:, 0, :]  # First element is the sequence output
        output = self.drop(logits)
        return self.out(output)
