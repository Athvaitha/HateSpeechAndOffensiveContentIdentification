import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration

# -----------------------------
# Load BLIP Model
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# -----------------------------
# Image Folder
# -----------------------------
image_folder = r"dataset/images"

captions = []

images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
print("Total image files:", len(images))
print(images[-5:])
for image_name in tqdm(images):

    path = os.path.join(image_folder, image_name)

    image = Image.open(path).convert("RGB")

    inputs = processor(images=image, text="a photo of a person", return_tensors="pt")

    output = model.generate(
        **inputs,
        num_beams=5,
        early_stopping=True,
        max_new_tokens=30
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    captions.append([image_name, caption])

df = pd.DataFrame(
    captions,
    columns=["ids", "caption"]
)

df.to_csv(
    "dataset/blip_captions.csv",
    index=False
)

print(df.head())