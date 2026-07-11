import pandas as pd

# Load training data
train = pd.read_csv(
    r"C:\Users\Dell\Documents\project\research_project\Telugu_HASOC\Telugu_HASOC\train_data_Telugu.csv"
)

# Load test data
test = pd.read_csv(
    r"C:\Users\Dell\Documents\project\research_project\Telugu_HASOC\Telugu_HASOC\test_data_Telugu.csv"
)

print("=" * 50)
print("TRAIN DATA")
print("=" * 50)

print(train.head())

print("\nColumns:")
print(train.columns)

print("\nShape:")
print(train.shape)

print("\nMissing Values:")
print(train.isnull().sum())

print("\nLabel Distribution:")
print(train.iloc[:, -1].value_counts())

print("\n")

print("=" * 50)
print("TEST DATA")
print("=" * 50)

print(test.head())

print("\nColumns:")
print(test.columns)

print("\nShape:")
print(test.shape)