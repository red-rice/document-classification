import torch
import torch.nn as nn
from transformers import AutoModel

class BertDocClassifier(nn.Module):
    """
    Outputs:
      - logits: [B, num_classes]
      - embedding h: [B, 768] from last hidden layer [CLS]
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768 for bert-base-uncased
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state       # [B, L, H]
        h = last_hidden[:, 0, :]                  # CLS token embedding [B, H]
        logits = self.classifier(h)
        return logits, h