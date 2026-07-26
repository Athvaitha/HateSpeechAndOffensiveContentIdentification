import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from model import MemeMultiTaskModel
from dataset import get_test_dataloader, INVERSE_LABEL_MAPPINGS

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_csv = "test.csv"
    best_model_path = "best_model.pt"
    output_path = "Tamil_HASOC/Tamil_HASOC/test_data_Tamil.csv"
    
    if not os.path.exists(best_model_path):
        print(f"Error: Trained model weights not found at {best_model_path}. Please train the model first.")
        sys.exit(1)
        
    print("Loading test dataset...")
    test_loader, image_names = get_test_dataloader(test_csv, batch_size=8)
    
    print("Initializing model and loading weights...")
    model = MemeMultiTaskModel()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    predictions = {
        "image_name": [],
        "predicted_sentiment": [],
        "predicted_sarcasm": [],
        "predicted_vulgarity": [],
        "predicted_abuse": [],
        "predicted_target": []
    }
    
    print("Running predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Extract task predictions
            pred_sentiment = torch.argmax(outputs["sentiment"], dim=-1).cpu().tolist()
            pred_sarcasm = torch.argmax(outputs["sarcasm"], dim=-1).cpu().tolist()
            pred_vulgarity = torch.argmax(outputs["vulgarity"], dim=-1).cpu().tolist()
            pred_abuse = torch.argmax(outputs["abuse"], dim=-1).cpu().tolist()
            pred_target = torch.argmax(outputs["target"], dim=-1).cpu().tolist()
            
            # Map predictions to label strings
            for idx in range(len(pred_sentiment)):
                predictions["predicted_sentiment"].append(INVERSE_LABEL_MAPPINGS["sentiment"][pred_sentiment[idx]])
                predictions["predicted_sarcasm"].append(INVERSE_LABEL_MAPPINGS["sarcasm"][pred_sarcasm[idx]])
                predictions["predicted_vulgarity"].append(INVERSE_LABEL_MAPPINGS["vulgarity"][pred_vulgarity[idx]])
                predictions["predicted_abuse"].append(INVERSE_LABEL_MAPPINGS["abuse"][pred_abuse[idx]])
                predictions["predicted_target"].append(INVERSE_LABEL_MAPPINGS["target"][pred_target[idx]])
                
    predictions["image_name"] = image_names
    
    # Save predictions
    df_pred = pd.DataFrame(predictions)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pred.to_csv(output_path, index=False)
    
    print(f"Predictions saved successfully to {output_path}!")
    print(df_pred.head())

if __name__ == "__main__":
    run_inference()
