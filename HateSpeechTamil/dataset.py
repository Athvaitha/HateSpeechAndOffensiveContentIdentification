import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

LABEL_MAPPINGS = {
    "sentiment": {"Negative": 0, "Neutral": 1, "Positive": 2},
    "sarcasm": {"Non-Sarcastic": 0, "Sarcastic": 1},
    "vulgarity": {"Non-Vulgar": 0, "Vulgar": 1},
    "abuse": {"Non-Abusive": 0, "Abusive": 1},
    "target": {"Person": 0, "Group": 1, "Organization": 2, "Other": 3, "None": 4}
}

INVERSE_LABEL_MAPPINGS = {
    task: {idx: label for label, idx in mapping.items()}
    for task, mapping in LABEL_MAPPINGS.items()
}

def load_and_prepare_dataset(csv_path, is_train=True):
    df = pd.read_csv(csv_path)
    df["gemini_output"] = df["gemini_output"].fillna("")
    if is_train:
        df["sentiment"] = df["sentiment"].fillna("Neutral")
        df["sarcasm"] = df["sarcasm"].fillna("Non-Sarcastic")
        df["vulgarity"] = df["vulgarity"].fillna("Non-Vulgar")
        df["abuse"] = df["abuse"].fillna("Non-Abusive")
        df["target"] = df["target"].fillna("None")
    return df

def get_dataloaders(train_csv, batch_size=8, validation_split=0.2, max_length=256, model_name="xlm-roberta-base"):
    """
    Loads train dataset, performs train-validation split, tokenizes text, and returns PyTorch DataLoader objects.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load raw data
    df = load_and_prepare_dataset(train_csv, is_train=True)
    
    # Map string labels to integers directly in pandas to avoid HuggingFace schema conflicts
    df["sentiment"] = df["sentiment"].map(LABEL_MAPPINGS["sentiment"])
    df["sarcasm"] = df["sarcasm"].map(LABEL_MAPPINGS["sarcasm"])
    df["vulgarity"] = df["vulgarity"].map(LABEL_MAPPINGS["vulgarity"])
    df["abuse"] = df["abuse"].map(LABEL_MAPPINGS["abuse"])
    df["target"] = df["target"].map(LABEL_MAPPINGS["target"])
    
    # Split into train/val
    train_df, val_df = train_test_split(
        df, 
        test_size=validation_split, 
        random_state=42, 
        stratify=df["target"]
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["gemini_output"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        
    # Apply tokenization
    train_mapped = train_dataset.map(tokenize_fn, batched=True)
    val_mapped = val_dataset.map(tokenize_fn, batched=True)
    
    # Set format to torch
    columns_to_keep = ["input_ids", "attention_mask", "sentiment", "sarcasm", "vulgarity", "abuse", "target"]
    train_mapped.set_format(type="torch", columns=columns_to_keep)
    val_mapped.set_format(type="torch", columns=columns_to_keep)
    
    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_mapped, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_mapped, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_test_dataloader(test_csv, batch_size=8, max_length=256, model_name="xlm-roberta-base"):
    """
    Loads test dataset, tokenizes text, and returns a PyTorch DataLoader object along with image names.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = load_and_prepare_dataset(test_csv, is_train=False)
    
    # Convert to HuggingFace Dataset
    test_dataset = Dataset.from_pandas(df)
    
    def tokenize_only(batch):
        return tokenizer(
            batch["gemini_output"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        
    test_mapped = test_dataset.map(tokenize_only, batched=True)
    test_mapped.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    test_loader = DataLoader(test_mapped, batch_size=batch_size, shuffle=False)
    return test_loader, df["image_name"].tolist()
