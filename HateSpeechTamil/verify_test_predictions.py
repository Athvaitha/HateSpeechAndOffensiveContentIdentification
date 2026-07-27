import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def verify_predictions():
    preds_path = "Tamil_HASOC/Tamil_HASOC/test_data_Tamil.csv"
    gt_path = "dataset/final_dataset.csv"
    
    if not os.path.exists(preds_path):
        print(f"Error: Predictions file not found at {preds_path}. Run inference.py first.")
        return
        
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at {gt_path}.")
        return
        
    print(f"Loading predictions from {preds_path}...")
    preds_df = pd.read_csv(preds_path)
    
    print(f"Loading ground truths from {gt_path}...")
    gt_df = pd.read_csv(gt_path)
    
    # Filter for test split
    test_gt = gt_df[gt_df["split"] == "test"].copy()
    
    # Standardize ground truth values to match predictions
    sentiment_map = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
    sarcasm_map = {"yes": "Sarcastic", "no": "Non-Sarcastic"}
    vulgarity_map = {"vulgar": "Vulgar", "not vulgar": "Non-Vulgar"}
    abuse_map = {"abusive": "Abusive", "non-abusive": "Non-Abusive"}
    target_map = {
        "individual": "Person",
        "social sub-groups": "Group",
        "gender": "Group",
        "political": "Organization",
        "others": "Other",
        "none": "None"
    }

    test_gt["sentiment"] = test_gt["sentiment"].map(sentiment_map)
    test_gt["sarcasm"] = test_gt["sarcasm"].map(sarcasm_map)
    test_gt["vulgarity"] = test_gt["vulgar"].map(vulgarity_map)
    test_gt["abuse"] = test_gt["abuse"].map(abuse_map)
    test_gt["target"] = test_gt["target"].map(target_map).fillna("None")
    
    # Merge datasets
    merged = preds_df.merge(test_gt, left_on="image_name", right_on="ids")
    
    print("\n" + "="*50)
    print("TEST SET EVALUATION METRICS (200 SAMPLES)")
    print("="*50)
    
    tasks = {
        "sentiment": ("predicted_sentiment", "sentiment"),
        "sarcasm": ("predicted_sarcasm", "sarcasm"),
        "vulgarity": ("predicted_vulgarity", "vulgarity"),
        "abuse": ("predicted_abuse", "abuse"),
        "target": ("predicted_target", "target")
    }
    
    for task_name, (pred_col, gt_col) in tasks.items():
        y_true = merged[gt_col].tolist()
        y_pred = merged[pred_col].tolist()
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        
        print(f"- {task_name.capitalize()}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  Macro F1:  {f1:.4f}")
        print()

if __name__ == "__main__":
    verify_predictions()
