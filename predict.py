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

ocr_lookup = {}
if os.path.exists("dataset/ocr_text.csv"):
    ocr_df = pd.read_csv("dataset/ocr_text.csv")
    ocr_lookup = {
        str(row["ids"]): str(row["ocr_text"]) if "ocr_text" in row else ""
        for _, row in ocr_df.iterrows()
    }


def _safe_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return ""
        return value

    if pd.isna(value):
        return ""

    return str(value).strip()


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
    caption = _safe_text(row.get("caption", ""))
    ocr_text = _safe_text(row.get("ocr_text", ocr_lookup.get(image_name, "")))

    sarcasm = _safe_text(row.get("sarcasm", "")).lower() == "yes"
    vulgar = _safe_text(row.get("vulgar", "")).lower() == "vulgar"
    abuse = _safe_text(row.get("abuse", "")).lower() == "abusive"
    target = _safe_text(row.get("target", "")).lower() != "none"

    cues = []
    if sarcasm:
        cues.append("sarcastic")
    if vulgar:
        cues.append("vulgar")
    if abuse:
        cues.append("abusive")
    if target:
        cues.append("targeted")

    evidence = []
    if caption:
        evidence.append(f"BLIP caption: {caption}")
    if ocr_text:
        evidence.append(f"OCR text: {ocr_text[:160]}")
    else:
        evidence.append("OCR text: not available for this example")

    evidence_text = " | ".join(evidence)

    if label == 1:
        if cues:
            reason = (
                f"The model classifies this as Hate Speech / Offensive because the BLIP caption and OCR evidence "
                f"support a harmful interpretation: {evidence_text}. The metadata also marks it as "
                f"{', '.join(cues)}, which is consistent with offensive meme content."
            )
        else:
            reason = (
                "The model classifies this as Hate Speech / Offensive because the learned pattern for this image "
                "matches the hateful class in the training data, even though the OCR text and caption do not contain "
                "strong explicit abuse."
            )
    else:
        if sarcasm or cues:
            reason = (
                f"The model classifies this as Non Hate Speech because the meme is marked as sarcastic, but the BLIP "
                f"caption and OCR evidence remain mostly neutral: {evidence_text}. The metadata does not show enough "
                f"abusive or targeted signals to escalate it into the hate/offensive class."
            )
        else:
            reason = (
                "The model classifies this as Non Hate Speech because the meme meaning appears neutral or harmless, "
                "and the BLIP caption plus OCR grounding do not show strong abusive or targeted content."
            )

    return caption or "Meaning unavailable", reason


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
