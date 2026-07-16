import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Configure stdout to support UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

image_path = r"dataset/images/image_tamil_0009.jpg"
if not os.path.exists(image_path):
    print(f"Error: sample image not found")
    sys.exit(1)

client = genai.Client()

# Test gemini-flash-lite-latest (which represents the fast, lightweight Flash Lite model)
test_model = "gemini-flash-lite-latest"

with open(image_path, "rb") as f:
    img_bytes = f.read()

print(f"Testing daily quota limit for model: {test_model}...")
success_count = 0

for i in range(1, 26):
    print(f"Request {i}/25...", end="", flush=True)
    try:
        response = client.models.generate_content(
            model=test_model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                "Generate a short caption."
            ]
        )
        print(" SUCCESS")
        success_count += 1
        time.sleep(2) # short delay to avoid RPM limits
    except Exception as e:
        print(f"\n FAILED on request {i}: {e}")
        break

print(f"\nCompleted: {success_count}/25 requests successful.")
if success_count == 25:
    print(f"Result: {test_model} supports more than 20 requests/day on this key!")
else:
    print(f"Result: {test_model} failed due to quota.")
