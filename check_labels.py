import pandas as pd

df = pd.read_csv("dataset/final_dataset.csv")

print(df["label"].value_counts())
print("\nPercentage:")
print(df["label"].value_counts(normalize=True) * 100)