import pandas as pd

train = pd.read_csv("dataset/train.csv")
captions = pd.read_csv("dataset/blip_captions.csv")

print("Train.csv rows:", len(train))
print("BLIP captions rows:", len(captions))

dataset = train.merge(
    captions,
    on="ids",
    how="left"
)

print("Merged rows:", len(dataset))

dataset.to_csv("dataset/final_dataset.csv", index=False)

print("Saved final_dataset.csv successfully!")