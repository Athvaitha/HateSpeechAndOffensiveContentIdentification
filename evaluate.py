import os
import sys
from PIL import Image

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Device
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -----------------------
# Load test dataset
# -----------------------

df = pd.read_csv("dataset/final_dataset.csv")

# Use the same test split seed as training for consistency
from sklearn.model_selection import train_test_split

_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------
# CLIP Processor
# -----------------------

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# -----------------------
# Dataset Class
# -----------------------

class MemeDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join("dataset", "images", row["ids"])
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

# -----------------------
# Model Definition
# -----------------------

class CLIPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        for param in self.clip.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)

        if isinstance(image_features, dict):
            image_features = image_features["pooler_output"]

        if hasattr(image_features, "last_hidden_state"):
            image_features = image_features.last_hidden_state[:, 0, :]

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        return self.classifier(image_features)

# -----------------------
# Load Model
# -----------------------

model_path = "models/best_model.pth"

if not os.path.exists(model_path):
    print("Model file not found:", model_path)
    print("Train first with: python train_model.py")
    sys.exit(1)

model = CLIPClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

test_dataset = MemeDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -----------------------
# Evaluate
# -----------------------

y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(pixel_values=pixel_values)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Model Evaluation")
print(classification_report(y_true, y_pred, zero_division=0))
print(confusion_matrix(y_true, y_pred))