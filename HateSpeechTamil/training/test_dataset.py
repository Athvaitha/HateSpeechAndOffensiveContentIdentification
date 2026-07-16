from dataset_loader import MemeDataset

dataset = MemeDataset(
    csv_file="dataset/final_dataset.csv",
    image_folder="dataset/images"
)

print("Total Samples:", len(dataset))

sample = dataset[0]

print()

print("Image Shape:")
print(sample["image"].shape)

print()

print("Input IDs Shape:")
print(sample["input_ids"].shape)

print()

print("Attention Mask Shape:")
print(sample["attention_mask"].shape)

print()

print("Label:")
print(sample["label"])