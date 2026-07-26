import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from model import MemeMultiTaskModel
from dataset import get_dataloaders
from utils import set_seed, compute_metrics_for_task

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    epochs = 5
    batch_size = 8
    lr = 2e-5
    
    # Load Data
    print("Loading data and preparing datasets...")
    train_loader, val_loader = get_dataloaders("train.csv", batch_size=batch_size)
    
    # Initialize Model
    print("Initializing model...")
    model = MemeMultiTaskModel()
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    start_epoch = 1
    checkpoint_path = "checkpoint.pt"
    
    # Load checkpoint if it exists to resume training
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at '{checkpoint_path}'. Loading and resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            print(f"Successfully loaded checkpoint. Resuming from Epoch {start_epoch}...")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}). Starting training from scratch.")
            
    print("Starting training...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training"):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            labels_sentiment = batch["sentiment"].to(device)
            labels_sarcasm = batch["sarcasm"].to(device)
            labels_vulgarity = batch["vulgarity"].to(device)
            labels_abuse = batch["abuse"].to(device)
            labels_target = batch["target"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss for each task
            loss_sentiment = criterion(outputs["sentiment"], labels_sentiment)
            loss_sarcasm = criterion(outputs["sarcasm"], labels_sarcasm)
            loss_vulgarity = criterion(outputs["vulgarity"], labels_vulgarity)
            loss_abuse = criterion(outputs["abuse"], labels_abuse)
            loss_target = criterion(outputs["target"], labels_target)
            
            # Total loss is the sum of all five losses
            total_loss = loss_sentiment + loss_sarcasm + loss_vulgarity + loss_abuse + loss_target
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        # Lists to store predictions and labels
        preds_all = {
            "sentiment": [],
            "sarcasm": [],
            "vulgarity": [],
            "abuse": [],
            "target": []
        }
        labels_all = {
            "sentiment": [],
            "sarcasm": [],
            "vulgarity": [],
            "abuse": [],
            "target": []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                labels_sentiment = batch["sentiment"].to(device)
                labels_sarcasm = batch["sarcasm"].to(device)
                labels_vulgarity = batch["vulgarity"].to(device)
                labels_abuse = batch["abuse"].to(device)
                labels_target = batch["target"].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                loss_sentiment = criterion(outputs["sentiment"], labels_sentiment)
                loss_sarcasm = criterion(outputs["sarcasm"], labels_sarcasm)
                loss_vulgarity = criterion(outputs["vulgarity"], labels_vulgarity)
                loss_abuse = criterion(outputs["abuse"], labels_abuse)
                loss_target = criterion(outputs["target"], labels_target)
                
                total_loss = loss_sentiment + loss_sarcasm + loss_vulgarity + loss_abuse + loss_target
                val_loss += total_loss.item()
                
                # Predictions
                for task in preds_all.keys():
                    preds_all[task].extend(torch.argmax(outputs[task], dim=-1).cpu().numpy())
                    
                labels_all["sentiment"].extend(labels_sentiment.cpu().numpy())
                labels_all["sarcasm"].extend(labels_sarcasm.cpu().numpy())
                labels_all["vulgarity"].extend(labels_vulgarity.cpu().numpy())
                labels_all["abuse"].extend(labels_abuse.cpu().numpy())
                labels_all["target"].extend(labels_target.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics for each task separately
        metrics = {}
        for task in preds_all.keys():
            metrics[task] = compute_metrics_for_task(preds_all[task], labels_all[task])
            
        print("\n" + "="*50)
        print(f"Epoch {epoch} Summary")
        print("="*50)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        print("\nValidation Metrics by Task:")
        for task, m in metrics.items():
            print(f"- {task.capitalize()}:")
            print(f"  Accuracy:  {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  Macro F1:  {m['f1']:.4f}")
            
        # Check validation loss and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"\n--> Saved new best model to best_model.pt with Val Loss: {best_val_loss:.4f}")
            
        # Save training checkpoint at the end of each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"--> Saved training checkpoint to '{checkpoint_path}' for Epoch {epoch}")
            
    # Clean up checkpoint on complete success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed temporary checkpoint file '{checkpoint_path}' since training is complete.")
        
    print("\nTraining complete!")

if __name__ == "__main__":
    train_model()
