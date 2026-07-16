import os
from PIL import Image

image_folder = r"dataset/images"

images = os.listdir(image_folder)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print(f"Total Images : {len(images)}")

extensions = {}

for img in images:
    ext = os.path.splitext(img)[1].lower()
    extensions[ext] = extensions.get(ext, 0) + 1

print("\nImage Formats:")
for k, v in extensions.items():
    print(f"{k} : {v}")

sample = os.path.join(image_folder, images[0])

image = Image.open(sample)

print("\nSample Image")

print("Filename :", images[0])
print("Size :", image.size)
print("Mode :", image.mode)