import torch
import torch.nn as nn
from transformers import AutoModel

class MemeMultiTaskModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout_prob=0.1):
        super().__init__()
        # Load the shared XLM-R encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size  # 768 for xlm-roberta-base
        
        # Shared dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Five separate classification heads
        self.sentiment_head = nn.Linear(hidden_size, 3)     # 3 classes: Negative, Neutral, Positive
        self.sarcasm_head = nn.Linear(hidden_size, 2)       # 2 classes: Non-Sarcastic, Sarcastic
        self.vulgarity_head = nn.Linear(hidden_size, 2)     # 2 classes: Non-Vulgar, Vulgar
        self.abuse_head = nn.Linear(hidden_size, 2)         # 2 classes: Non-Abusive, Abusive
        self.target_head = nn.Linear(hidden_size, 5)        # 5 classes: Person, Group, Organization, Other, None
        
    def forward(self, input_ids, attention_mask):
        # Pass input through the encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract sentence representation using the [CLS] token (first token at index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        
        # Pass through individual heads to get raw logits
        sentiment_logits = self.sentiment_head(cls_output)
        sarcasm_logits = self.sarcasm_head(cls_output)
        vulgarity_logits = self.vulgarity_head(cls_output)
        abuse_logits = self.abuse_head(cls_output)
        target_logits = self.target_head(cls_output)
        
        return {
            "sentiment": sentiment_logits,
            "sarcasm": sarcasm_logits,
            "vulgarity": vulgarity_logits,
            "abuse": abuse_logits,
            "target": target_logits
        }
