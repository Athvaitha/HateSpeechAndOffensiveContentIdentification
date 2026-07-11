import pandas as pd

df = pd.read_csv("dataset/final_dataset.csv")

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())