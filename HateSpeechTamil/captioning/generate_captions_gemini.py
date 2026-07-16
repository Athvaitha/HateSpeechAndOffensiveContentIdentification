import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load and force override environment variables from .env file
load_dotenv(override=True)

from google import genai
from google.genai import types

# Configure stdout to support UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Directories and paths
image_folder = r"dataset/images"
train_csv_path = "dataset/train.csv"
gemini_csv_path = "dataset/blip_captions_gemini.csv"
final_csv_path = "dataset/blip_captions.csv"
backup_csv_path = "dataset/blip_captions_bak.csv"

# Fetch API Key explicitly
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment or .env file.")
    sys.exit(1)

# Initialize Gemini Client with the specific key
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    sys.exit(1)

# Load list of target images from train.csv to maintain dataset alignment
if not os.path.exists(train_csv_path):
    print(f"Error: {train_csv_path} not found. Reading files directly from {image_folder} instead...")
    images = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])
else:
    train_df = pd.read_csv(train_csv_path)
    images = train_df["ids"].tolist()

print(f"Total target images to process: {len(images)}")

# Load existing progress if available
processed_data = {}
if os.path.exists(gemini_csv_path):
    try:
        df_existing = pd.read_csv(gemini_csv_path)
        df_existing = df_existing.drop_duplicates(subset=["ids"])
        processed_data = dict(zip(df_existing["ids"], df_existing["caption"]))
        print(f"Loaded existing progress. {len(processed_data)} images already processed.")
    except Exception as e:
        print(f"Error loading existing progress: {e}. Starting fresh.")
        processed_data = {}

# Process remaining images
try:
    for image_name in tqdm(images):
        if image_name in processed_data and pd.notna(processed_data[image_name]) and str(processed_data[image_name]).strip():
            continue

        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found at {image_path}. Skipping.")
            continue

        # Determine MIME type dynamically
        mime_type = "image/jpeg"
        if image_name.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_name.lower().endswith(".webp"):
            mime_type = "image/webp"

        # Read image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        success = False
        retries = 3
        while not success and retries > 0:
            try:
                # Generate caption using Gemini 1.5/2.0 Flash Lite (via gemini-flash-lite-latest alias)
                response = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        ),
                        "Generate a concise, single-paragraph descriptive caption for this meme. Explain the visual context, facial expressions, and translate any Tamil text overlay into English. Do not use formatting like bullet points or bold text."
                    ]
                )
                
                # Retrieve and clean text
                caption = response.text.strip().replace("\n", " ").replace("\r", "")
                processed_data[image_name] = caption
                success = True
                
                # Print output to confirm it's working
                print(f"\n[Success] {image_name}: {caption[:100]}...")
                
            except Exception as e:
                retries -= 1
                print(f"\nError processing {image_name}: {e}. Retries remaining: {retries}")
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    print("Rate limit hit. Waiting 15 seconds before retry...")
                    time.sleep(15)
                else:
                    time.sleep(2)

        # Write progress incrementally to prevent data loss
        if success:
            df_temp = pd.DataFrame(list(processed_data.items()), columns=["ids", "caption"])
            df_temp.to_csv(gemini_csv_path, index=False)
            
            # Sleep 5 seconds to stay safely under the 15 RPM (Requests Per Minute) free tier limit
            time.sleep(5)

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving current progress...")
    df_temp = pd.DataFrame(list(processed_data.items()), columns=["ids", "caption"])
    df_temp.to_csv(gemini_csv_path, index=False)
    sys.exit(0)

# Check if all images were processed successfully
all_done = all(img in processed_data for img in images)
if all_done:
    print("\nAll images processed successfully!")
    
    # Backup original blip_captions.csv if it exists
    if os.path.exists(final_csv_path):
        if os.path.exists(backup_csv_path):
            os.remove(backup_csv_path)
        os.rename(final_csv_path, backup_csv_path)
        print(f"Backed up original blip_captions.csv to {backup_csv_path}")
        
    # Move the new gemini captions to the final path
    os.rename(gemini_csv_path, final_csv_path)
    print(f"Replaced blip_captions.csv with new Gemini-generated captions at {final_csv_path}!")
else:
    print(f"\nProgress saved. Processed {len(processed_data)}/{len(images)} images. Run script again to resume.")
