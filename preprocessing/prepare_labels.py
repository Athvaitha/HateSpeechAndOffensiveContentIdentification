import pandas as pd

train = pd.read_csv(
    r"C:\Users\Dell\Documents\project\research_project\Telugu_HASOC\Telugu_HASOC\train_data_Telugu.csv"
)

train["label"] = train["abuse"].map({
    "non-abusive": 0,
    "abusive": 1
})

print(train[["ids", "abuse", "label"]].head())

train.to_csv("dataset/train.csv", index=False)

print("Dataset saved!")