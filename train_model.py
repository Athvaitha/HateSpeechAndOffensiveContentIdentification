import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

os.makedirs("models", exist_ok=True)

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

        encoding = processor(
            images=image,
            return_tensors="pt"
        )

        return {
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

        image_features = self.clip.get_image_features(
            pixel_values=pixel_values
        )

        if isinstance(image_features, dict):
            image_features = image_features["pooler_output"]

        if hasattr(image_features, "last_hidden_state"):
            image_features = image_features.last_hidden_state[:, 0, :]

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        logits = self.classifier(image_features)

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

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values)

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

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(pixel_values=pixel_values)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    acc = accuracy_score(actuals, predictions)
    pre = precision_score(actuals, predictions, zero_division=0)
    rec = recall_score(actuals, predictions, zero_division=0)
    f1 = f1_score(actuals, predictions, zero_division=0)
    cm = confusion_matrix(actuals, predictions)

    return acc, pre, rec, f1, cm


# -----------------------
# Train Model
# -----------------------

num_epochs = 3
best_loss = float("inf")

for epoch in range(num_epochs):

    loss = train_one_epoch()

    acc, pre, rec, f1, cm = evaluate()

    print("=" * 50)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("=" * 50)

    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Best model saved as models/best_model.pth")

print("Training complete. Best checkpoint saved at models/best_model.pth")