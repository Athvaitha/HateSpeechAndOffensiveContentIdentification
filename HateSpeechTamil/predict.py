import os
import sys
import torch
import torch.nn as nn
from PIL import Image
# from paddleocr import PaddleOCR
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv(override=True)

# Configure stdout to support UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# -----------------------
# Initialize Models (Cached locally)
# -----------------------
# print("Initializing PaddleOCR...")
# ocr = PaddleOCR(use_angle_cls=True, lang="ta", enable_mkldnn=False)

print("Initializing Gemini client...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment or .env file.")
    sys.exit(1)
try:
    gemini_client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    sys.exit(1)

print("Initializing CLIP processor & model...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# -----------------------
# Define Model Architecture
# -----------------------
class MemeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = clip_model
        self.sentiment_classifier = nn.Linear(1024, 3)
        self.sarcasm_classifier = nn.Linear(1024, 2)
        self.vulgar_classifier = nn.Linear(1024, 2)
        self.abuse_classifier = nn.Linear(1024, 2)
        self.target_classifier = nn.Linear(1024, 5)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        features = torch.cat((image_features, text_features), dim=1)
        
        logits_sentiment = self.sentiment_classifier(features)
        logits_sarcasm = self.sarcasm_classifier(features)
        logits_vulgar = self.vulgar_classifier(features)
        logits_abuse = self.abuse_classifier(features)
        logits_target = self.target_classifier(features)
        return logits_sentiment, logits_sarcasm, logits_vulgar, logits_abuse, logits_target

# -----------------------
# Load Model Weights
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MemeClassifier().to(device)

best_model_path = "best_model.pt"
if os.path.exists(best_model_path):
    print(f"Loading trained weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
else:
    print("Warning: best_model.pt not found. Predictions will use untrained weights.")

model.eval()

# -----------------------
# Predict Function
# -----------------------
def predict_meme(image_path):
    if not os.path.exists(image_path):
        return {"error": f"Image path '{image_path}' does not exist."}

    # 1. Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    # OCR is skipped since the Gemini description translates visual and text overlays into English
    text = "Skipped (Using Gemini description instead)"

    # 3. Generate English scene caption using Gemini
    print("Running Gemini scene description...")
    try:
        # Determine MIME type dynamically
        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith(".webp"):
            mime_type = "image/webp"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = gemini_client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                ),
                "Generate a concise, single-paragraph descriptive caption for this meme. Explain the visual context, facial expressions, and translate any Tamil text overlay into English. Do not use formatting like bullet points or bold text."
            ]
        )
        caption = response.text.strip().replace("\n", " ").replace("\r", "")
    except Exception as e:
        print(f"Gemini Captioning Error: {e}")
        caption = ""

    # Use strictly the Gemini caption for CLIP input
    combined_text = caption

    # 5. Tokenize/Process for CLIP model
    print("Running prediction through model...")
    inputs = clip_processor(
        text=[combined_text],
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    # 6. Forward pass
    with torch.no_grad():
        logits_sentiment, logits_sarcasm, logits_vulgar, logits_abuse, logits_target = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # Sentiment prediction
        probs_sentiment = torch.softmax(logits_sentiment, dim=1)[0]
        sentiment_idx = torch.argmax(probs_sentiment).item()
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_label = sentiment_map[sentiment_idx]
        sentiment_conf = probs_sentiment[sentiment_idx].item()
        
        # Sarcasm prediction
        probs_sarcasm = torch.softmax(logits_sarcasm, dim=1)[0]
        sarcasm_idx = torch.argmax(probs_sarcasm).item()
        sarcasm_map = {0: "Non-Sarcastic", 1: "Sarcastic"}
        sarcasm_label = sarcasm_map[sarcasm_idx]
        sarcasm_conf = probs_sarcasm[sarcasm_idx].item()
        
        # Vulgarity prediction
        probs_vulgar = torch.softmax(logits_vulgar, dim=1)[0]
        vulgar_idx = torch.argmax(probs_vulgar).item()
        vulgar_map = {0: "Not Vulgar", 1: "Vulgar"}
        vulgar_label = vulgar_map[vulgar_idx]
        vulgar_conf = probs_vulgar[vulgar_idx].item()
        
        # Abuse prediction
        probs_abuse = torch.softmax(logits_abuse, dim=1)[0]
        abuse_idx = torch.argmax(probs_abuse).item()
        abuse_map = {0: "Non-abusive", 1: "Abusive"}
        abuse_label = abuse_map[abuse_idx]
        abuse_conf = probs_abuse[abuse_idx].item()
        
        # Target prediction
        probs_target = torch.softmax(logits_target, dim=1)[0]
        target_idx = torch.argmax(probs_target).item()
        target_map = {0: "Person", 1: "Group", 2: "Organization", 3: "Other", 4: "None"}
        target_label = target_map[target_idx]
        target_conf = probs_target[target_idx].item()
        
        # Determine active multi-label tags
        multi_label = []
        if abuse_label == "Abusive":
            multi_label.append("Abuse")
        if sarcasm_label == "Sarcastic":
            multi_label.append("Sarcasm")
        if vulgar_label == "Vulgar":
            multi_label.append("Vulgarity")
        if target_label != "None":
            multi_label.append("Target")

    return {
        "ocr_text": text,
        "caption": caption,
        "sentiment": sentiment_label,
        "sentiment_confidence": sentiment_conf,
        "sarcasm": sarcasm_label,
        "sarcasm_confidence": sarcasm_conf,
        "vulgarity": vulgar_label,
        "vulgarity_confidence": vulgar_conf,
        "abuse": abuse_label,
        "abuse_confidence": abuse_conf,
        "target": target_label,
        "target_confidence": target_conf,
        "multi_label": multi_label,
        # Keep old prediction field for backup/minimal compatibility
        "prediction": abuse_label, 
        "confidence": abuse_conf
    }

# -----------------------
# Command Line Interface
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    res = predict_meme(img_path)

    if "error" in res:
        print(f"Error: {res['error']}")
    else:
        print("\n" + "=" * 50)
        print("             INFERENCE RESULTS             ")
        print("=" * 50)
        print(f"Image Path:          {img_path}")
        print(f"Extracted OCR Text:  {res['ocr_text']}")
        print(f"Visual Description:  {res['caption']}")
        print("-" * 50)
        print(f"Sentiment:           {res['sentiment']} ({res['sentiment_confidence']:.2%})")
        print(f"Sarcasm:             {res['sarcasm']} ({res['sarcasm_confidence']:.2%})")
        print(f"Vulgarity:           {res['vulgarity']} ({res['vulgarity_confidence']:.2%})")
        print(f"Abuse:               {res['abuse']} ({res['abuse_confidence']:.2%})")
        print(f"Target:              {res['target']} ({res['target_confidence']:.2%})")
        print(f"Multi-label:         {', '.join(res['multi_label']) if res['multi_label'] else 'None'}")
        print("=" * 50)
