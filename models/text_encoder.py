import torch
import torch.nn as nn

from transformers import AutoModel


class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(
    "distilbert-base-multilingual-cased",
    cache_dir="D:/HF_CACHE"
)

    def forward(self, input_ids, attention_mask):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0,:]

        return cls