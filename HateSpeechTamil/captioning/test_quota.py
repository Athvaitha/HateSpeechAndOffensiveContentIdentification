import os
import sys
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Configure stdout to support UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

image_path = r"dataset/images/image_tamil_0009.jpg"
if not os.path.exists(image_path):
    print("Error: sample image not found")
    sys.exit(1)

client = genai.Client()

# List of visual models to test
candidate_models = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001"
]

with open(image_path, "rb") as f:
    img_bytes = f.read()

print("Testing alternative models for active quota...")
for model_name in candidate_models:
    print(f"Testing {model_name}...", end="", flush=True)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                "Generate a short caption."
            ]
        )
        print(" SUCCESS!")
        print(f"  Response: {response.text.strip()}")
        print(f"  --> FOUND A WORKING MODEL: {model_name}\n")
    except Exception as e:
        print(f" FAILED: {str(e)[:150]}...")

print("Test complete.")
