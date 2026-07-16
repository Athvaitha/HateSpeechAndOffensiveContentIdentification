import os
import pandas as pd

# Load Tamil train and test datasets
train = pd.read_csv("Tamil_HASOC/Tamil_HASOC/train_data_Tamil.csv")
test = pd.read_csv("Tamil_HASOC/Tamil_HASOC/test_data_Tamil.csv")

# Map labels to binary
train["label"] = train["abuse"].map({
    "non-abusive": 0,
    "abusive": 1
})
test["label"] = test["abuse"].map({
    "non-abusive": 0,
    "abusive": 1
})

# Add split column
train["split"] = "train"
test["split"] = "test"

# Combine datasets
combined = pd.concat([train, test], ignore_index=True)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Combined shape:", combined.shape)
print("\nLabel distribution:")
print(combined["label"].value_counts())

# Save to dataset/train.csv
os.makedirs("dataset", exist_ok=True)
combined.to_csv("dataset/train.csv", index=False)

print("\nDataset saved to dataset/train.csv!")