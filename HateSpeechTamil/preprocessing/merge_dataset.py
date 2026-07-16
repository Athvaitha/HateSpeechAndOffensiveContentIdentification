import pandas as pd

train = pd.read_csv("dataset/train.csv")
captions = pd.read_csv("dataset/blip_captions.csv")
ocr = pd.read_csv("dataset/ocr_text.csv")

# Merge captions
dataset = train.merge(
    captions,
    on="ids",
    how="left"
)

# Merge OCR text
dataset = dataset.merge(
    ocr,
    on="ids",
    how="left"
)

dataset.to_csv(
    "dataset/final_dataset.csv",
    index=False
)

# Reconfigure stdout to support UTF-8 encoding on Windows console
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print(dataset.head())

print()

print("Shape:", dataset.shape)

print()

print(dataset.isnull().sum())