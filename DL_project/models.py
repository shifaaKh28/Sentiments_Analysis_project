from torch import nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, pre_trained_model_name):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pre_trained_model_name)  # Pre-trained model
        self.dropout = nn.Dropout(p=0.3)  # Dropout to reduce overfitting
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # Output layer

    def forward(self, input_ids, attention_mask):
        # Forward pass through the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        cls_output = outputs[0][:, 0, :]  # Use the representation of the [CLS] token
        cls_output = self.dropout(cls_output)
        return self.out(cls_output)
