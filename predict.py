import os
import sys
from PIL import Image

import pandas as pd
import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

# -----------------------
# Device
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# -----------------------
# Load CLIP Processor
# -----------------------

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# -----------------------
# Define CLIP Classifier Model
# -----------------------

class CLIPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        for param in self.clip.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values):

        image_features = self.clip.get_image_features(
            pixel_values=pixel_values
        )

        if isinstance(image_features, dict):
            image_features = image_features["pooler_output"]

        if hasattr(image_features, "last_hidden_state"):
            image_features = image_features.last_hidden_state[:, 0, :]

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        output = self.classifier(image_features)

        return output


# -----------------------
# Load Metadata for Explanation
# -----------------------

metadata_df = pd.read_csv("dataset/final_dataset.csv")

# -----------------------
# Load Trained Model
# -----------------------

model_path = "models/best_model.pth"

if not os.path.exists(model_path):
    print("Model file not found:", model_path)
    print("Train first with: python train_model.py")
    sys.exit(1)

model = CLIPClassifier()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model Loaded Successfully")


# -----------------------
# Prediction Function
# -----------------------

def get_meme_meaning_and_reason(image_name, label):

    row = metadata_df[metadata_df["ids"] == image_name]

    if row.empty:
        return (
            "Meaning not available in dataset metadata.",
            "The model was still used for prediction, but there is no matching dataset row to explain the meme meaning."
        )

    row = row.iloc[0]
    caption = str(row["caption"]).strip()

    sarcasm = str(row["sarcasm"]).strip().lower() == "yes"
    vulgar = str(row["vulgar"]).strip().lower() == "vulgar"
    abuse = str(row["abuse"]).strip().lower() == "abusive"
    target = str(row["target"]).strip().lower() != "none"

    cues = []
    if sarcasm:
        cues.append("sarcastic")
    if vulgar:
        cues.append("vulgar")
    if abuse:
        cues.append("abusive")
    if target:
        cues.append("targeted")

    if label == 1:
        if cues:
            reason = (
                f"The model classifies this as Hate Speech / Offensive because the dataset metadata marks it with "
                f"{', '.join(cues)} cues, which are consistent with offensive meme content."
            )
        else:
            reason = (
                "The model classifies this as Hate Speech / Offensive because the learned pattern for this image "
                "matches the hateful class in the training data."
            )
    else:
        if cues:
            reason = (
                f"The model classifies this as Non Hate Speech because the meme text and metadata are mostly neutral, "
                f"and the harmful cues are not strong enough to trigger the hate/offensive class."
            )
        else:
            reason = (
                "The model classifies this as Non Hate Speech because the meme meaning appears neutral or harmless, "
                "and its metadata does not show strong abusive or targeted signals."
            )

    return caption, reason


def predict(image_path):

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")
    image_name = os.path.basename(image_path)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():

        outputs = model(pixel_values)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        )

    label = prediction.item()
    confidence = probabilities[0][label].item() * 100

    if label == 1:
        result = "Hate Speech / Offensive"
    else:
        result = "Non Hate Speech"

    caption, reason = get_meme_meaning_and_reason(image_name, label)

    print("\nPrediction Result")
    print("-----------------")
    print("Class:", result)
    print("Confidence: {:.2f}%".format(confidence))
    print("Meaning of the meme:", caption)
    print("Why this result:", reason)


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:\npython predict.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]
    predict(image_path)
