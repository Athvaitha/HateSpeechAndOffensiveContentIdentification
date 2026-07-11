import pandas as pd

train = pd.read_csv("dataset/train.csv")

captions = pd.read_csv("dataset/blip_captions.csv")

dataset = train.merge(
    captions,
    on="ids",
    how="left"
)

dataset.to_csv(
    "dataset/final_dataset.csv",
    index=False
)

print(dataset.head())

print()

print("Shape:", dataset.shape)

print()

print(dataset.isnull().sum())