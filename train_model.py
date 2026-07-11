import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------
# Device
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("dataset/final_dataset.csv")

print(df.head())
print("Total Samples:", len(df))

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train:", len(train_df))
print("Test :", len(test_df))

# -----------------------
# Load CLIP Processor
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

        image_path = os.path.join(
            "dataset",
            "images",
            row["ids"]
        )

        image = Image.open(image_path).convert("RGB")

        text = row["caption"]

        encoding = processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

train_dataset = MemeDataset(train_df)
test_dataset = MemeDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print("Dataset Loaded Successfully")
# -----------------------
# CLIP Model + Classifier
# -----------------------

class MemeClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained CLIP
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values):

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # Combine image + text features
        features = torch.cat(
            (image_features, text_features),
            dim=1
        )

        logits = self.classifier(features)

        return logits


# -----------------------
# Initialize Model
# -----------------------

model = MemeClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

print("Model Loaded Successfully")
# -----------------------
# Training Function
# -----------------------

def train_one_epoch():

    model.train()

    running_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# -----------------------
# Evaluation Function
# -----------------------

def evaluate():

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    acc = accuracy_score(actuals, predictions)
    pre = precision_score(actuals, predictions)
    rec = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)

    return acc, pre, rec, f1


# -----------------------
# Train Model
# -----------------------

epochs = 3

for epoch in range(epochs):

    loss = train_one_epoch()

    acc, pre, rec, f1 = evaluate()

    print("=" * 50)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 50)