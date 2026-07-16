import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from torchvision import transforms


class MemeDataset(Dataset):

    def __init__(self, csv_file, image_folder):

        # Load CSV
        self.data = pd.read_csv(csv_file)

        # Image folder
        self.image_folder = image_folder

        # DistilBERT Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-multilingual-cased",
            cache_dir="D:/HF_CACHE"
        )

        # Image Transform (ResNet50)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # ------------------------
        # Load Image
        # ------------------------
        image_path = os.path.join(
            self.image_folder,
            row["ids"]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # ------------------------
        # OCR + Caption
        # ------------------------
        ocr_text = str(row["ocr_text"]) if "ocr_text" in row else ""
        caption = str(row["caption"]) if "caption" in row else ""

        combined_text = ocr_text + " [SEP] " + caption

        # ------------------------
        # Tokenize
        # ------------------------
        encoding = self.tokenizer(
            combined_text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # ------------------------
        # Label
        # ------------------------
        if "label" in row:
            label = int(row["label"])
        else:
            label = -1

        return {

            "image": image,

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "label":
                torch.tensor(label, dtype=torch.long)

        }