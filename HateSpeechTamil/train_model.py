import os
import sys
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Configure stdout to support UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# -----------------------
# Device Setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("dataset/final_dataset.csv")
print("Total Samples in Dataset:", len(df))

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

# Oversample minority class (label 1) in train_df to handle extreme class imbalance
train_df_majority = train_df[train_df["label"] == 0]
train_df_minority = train_df[train_df["label"] == 1]
if len(train_df_minority) > 0:
    train_df_minority_oversampled = train_df_minority.sample(len(train_df_majority), replace=True, random_state=42)
    train_df = pd.concat([train_df_majority, train_df_minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Oversampled Train samples:", len(train_df))
print("Oversampled Train label counts:\n", train_df["label"].value_counts())
print("Test samples :", len(test_df))
# -----------------------
# Load CLIP Processor & Model Setup
# -----------------------
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def map_sentiment(val):
    val = str(val).lower().strip()
    if val == "negative": return 0
    if val == "neutral": return 1
    if val == "positive": return 2
    return 1

def map_sarcasm(val):
    val = str(val).lower().strip()
    if val == "yes": return 1
    return 0

def map_vulgar(val):
    val = str(val).lower().strip()
    if val == "vulgar": return 1
    return 0

def map_abuse(val):
    val = str(val).lower().strip()
    if val == "abusive": return 1
    return 0

def map_target(val):
    val = str(val).lower().strip()
    if val == "individual": return 0
    if val in ["social sub-groups", "gender"]: return 1
    if val == "political": return 2
    if val == "others": return 3
    return 4

class MemeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP parameters to preserve pre-trained embeddings
        for param in self.clip.parameters():
            param.requires_grad = False

        self.sentiment_classifier = nn.Linear(1024, 3)
        self.sarcasm_classifier = nn.Linear(1024, 2)
        self.vulgar_classifier = nn.Linear(1024, 2)
        self.abuse_classifier = nn.Linear(1024, 2)
        self.target_classifier = nn.Linear(1024, 5)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        features = torch.cat((image_features, text_features), dim=1)
        logits_sentiment = self.sentiment_classifier(features)
        logits_sarcasm = self.sarcasm_classifier(features)
        logits_vulgar = self.vulgar_classifier(features)
        logits_abuse = self.abuse_classifier(features)
        logits_target = self.target_classifier(features)
        return logits_sentiment, logits_sarcasm, logits_vulgar, logits_abuse, logits_target

model = MemeClassifier().to(device)

# -----------------------
# Feature Caching Pipeline
# -----------------------
def extract_features(dataframe, desc):
    print(f"\nCaching CLIP features for: {desc}...")
    features_list = []
    labels_sentiment = []
    labels_sarcasm = []
    labels_vulgar = []
    labels_abuse = []
    labels_target = []
    
    # Temporarily set model to eval to prevent dropout, etc.
    model.eval()
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        image_path = os.path.join("dataset", "images", row["ids"])
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} does not exist.")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {row['ids']}: {e}")
            continue
            
        caption = str(row["caption"]) if ("caption" in row and pd.notna(row["caption"])) else ""
        
        with torch.no_grad():
            inputs = processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.clip(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            combined = torch.cat((image_features, text_features), dim=1)
            
            features_list.append(combined.cpu())
            labels_sentiment.append(map_sentiment(row.get("sentiment", "neutral")))
            labels_sarcasm.append(map_sarcasm(row.get("sarcasm", "no")))
            labels_vulgar.append(map_vulgar(row.get("vulgar", "not vulgar")))
            labels_abuse.append(map_abuse(row.get("abuse", "non-abusive")))
            labels_target.append(map_target(row.get("target", "none")))
            
    return (
        torch.cat(features_list, dim=0),
        torch.tensor(labels_sentiment, dtype=torch.long),
        torch.tensor(labels_sarcasm, dtype=torch.long),
        torch.tensor(labels_vulgar, dtype=torch.long),
        torch.tensor(labels_abuse, dtype=torch.long),
        torch.tensor(labels_target, dtype=torch.long)
    )

train_features, train_sentiment, train_sarcasm, train_vulgar, train_abuse, train_target = extract_features(train_df, "Train Set")
test_features, test_sentiment, test_sarcasm, test_vulgar, test_abuse, test_target = extract_features(test_df, "Test Set")

# -----------------------
# Cached Dataloaders
# -----------------------
class CachedDataset(Dataset):
    def __init__(self, features, label_sentiment, label_sarcasm, label_vulgar, label_abuse, label_target):
        self.features = features
        self.label_sentiment = label_sentiment
        self.label_sarcasm = label_sarcasm
        self.label_vulgar = label_vulgar
        self.label_abuse = label_abuse
        self.label_target = label_target
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.label_sentiment[idx],
            self.label_sarcasm[idx],
            self.label_vulgar[idx],
            self.label_abuse[idx],
            self.label_target[idx]
        )

train_loader = DataLoader(CachedDataset(train_features, train_sentiment, train_sarcasm, train_vulgar, train_abuse, train_target), batch_size=16, shuffle=True)
test_loader = DataLoader(CachedDataset(test_features, test_sentiment, test_sarcasm, test_vulgar, test_abuse, test_target), batch_size=16, shuffle=False)

print("\nDataset Loaded Successfully and Cached!")

# -----------------------
# Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.1
)

# -----------------------
# Train & Evaluate Loops
# -----------------------
def train_one_epoch():
    model.sentiment_classifier.train()
    model.sarcasm_classifier.train()
    model.vulgar_classifier.train()
    model.abuse_classifier.train()
    model.target_classifier.train()
    
    running_loss = 0
    for batch_x, batch_sentiment, batch_sarcasm, batch_vulgar, batch_abuse, batch_target in train_loader:
        batch_x = batch_x.to(device)
        batch_sentiment = batch_sentiment.to(device)
        batch_sarcasm = batch_sarcasm.to(device)
        batch_vulgar = batch_vulgar.to(device)
        batch_abuse = batch_abuse.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        
        out_sentiment = model.sentiment_classifier(batch_x)
        out_sarcasm = model.sarcasm_classifier(batch_x)
        out_vulgar = model.vulgar_classifier(batch_x)
        out_abuse = model.abuse_classifier(batch_x)
        out_target = model.target_classifier(batch_x)
        
        loss_sentiment = criterion(out_sentiment, batch_sentiment)
        loss_sarcasm = criterion(out_sarcasm, batch_sarcasm)
        loss_vulgar = criterion(out_vulgar, batch_vulgar)
        loss_abuse = criterion(out_abuse, batch_abuse)
        loss_target = criterion(out_target, batch_target)
        
        loss = loss_sentiment + loss_sarcasm + loss_vulgar + loss_abuse + loss_target
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate():
    model.sentiment_classifier.eval()
    model.sarcasm_classifier.eval()
    model.vulgar_classifier.eval()
    model.abuse_classifier.eval()
    model.target_classifier.eval()
    
    y_true_sentiment, y_pred_sentiment = [], []
    y_true_sarcasm, y_pred_sarcasm = [], []
    y_true_vulgar, y_pred_vulgar = [], []
    y_true_abuse, y_pred_abuse = [], []
    y_true_target, y_pred_target = [], []
    
    with torch.no_grad():
        for batch_x, batch_sentiment, batch_sarcasm, batch_vulgar, batch_abuse, batch_target in test_loader:
            batch_x = batch_x.to(device)
            
            out_sentiment = model.sentiment_classifier(batch_x)
            out_sarcasm = model.sarcasm_classifier(batch_x)
            out_vulgar = model.vulgar_classifier(batch_x)
            out_abuse = model.abuse_classifier(batch_x)
            out_target = model.target_classifier(batch_x)
            
            y_pred_sentiment.extend(torch.argmax(out_sentiment, dim=1).cpu().numpy())
            y_pred_sarcasm.extend(torch.argmax(out_sarcasm, dim=1).cpu().numpy())
            y_pred_vulgar.extend(torch.argmax(out_vulgar, dim=1).cpu().numpy())
            y_pred_abuse.extend(torch.argmax(out_abuse, dim=1).cpu().numpy())
            y_pred_target.extend(torch.argmax(out_target, dim=1).cpu().numpy())
            
            y_true_sentiment.extend(batch_sentiment.numpy())
            y_true_sarcasm.extend(batch_sarcasm.numpy())
            y_true_vulgar.extend(batch_vulgar.numpy())
            y_true_abuse.extend(batch_abuse.numpy())
            y_true_target.extend(batch_target.numpy())
            
    acc_sentiment = accuracy_score(y_true_sentiment, y_pred_sentiment)
    acc_sarcasm = accuracy_score(y_true_sarcasm, y_pred_sarcasm)
    acc_vulgar = accuracy_score(y_true_vulgar, y_pred_vulgar)
    acc_abuse = accuracy_score(y_true_abuse, y_pred_abuse)
    acc_target = accuracy_score(y_true_target, y_pred_target)
    
    f1_sentiment = f1_score(y_true_sentiment, y_pred_sentiment, average='macro', zero_division=0)
    f1_sarcasm = f1_score(y_true_sarcasm, y_pred_sarcasm, average='macro', zero_division=0)
    f1_vulgar = f1_score(y_true_vulgar, y_pred_vulgar, average='macro', zero_division=0)
    f1_abuse = f1_score(y_true_abuse, y_pred_abuse, average='macro', zero_division=0)
    f1_target = f1_score(y_true_target, y_pred_target, average='macro', zero_division=0)
    
    metrics = {
        'sentiment': (acc_sentiment, f1_sentiment),
        'sarcasm': (acc_sarcasm, f1_sarcasm),
        'vulgarity': (acc_vulgar, f1_vulgar),
        'abuse': (acc_abuse, f1_abuse),
        'target': (acc_target, f1_target)
    }
    return metrics

# -----------------------
# Training Loop
# -----------------------
epochs = 15
best_avg_f1 = 0.0

print("\nStarting Training...")
for epoch in range(epochs):
    loss = train_one_epoch()
    metrics = evaluate()
    
    avg_f1 = sum(m[1] for m in metrics.values()) / len(metrics)
    
    print("=" * 60)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Avg F1: {avg_f1:.4f}")
    for task, (acc, f1) in metrics.items():
        print(f"  - {task.capitalize():<12}: Acc = {acc:.4f}, Macro-F1 = {f1:.4f}")
    print("=" * 60)

    # Save the best model based on average F1
    if avg_f1 >= best_avg_f1:
        best_avg_f1 = avg_f1
        torch.save(model.state_dict(), "best_model.pt")
        print(f"--> Saved new best model to best_model.pt (Avg F1 Score: {best_avg_f1:.4f})")

print("\nTraining completed successfully!")