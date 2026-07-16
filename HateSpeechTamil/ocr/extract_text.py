import os
import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

# Initialize OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ta",     # Tamil
    enable_mkldnn=False
)

image_folder = r"dataset/images"
csv_path = "dataset/ocr_text.csv"

# Load existing progress if available
if os.path.exists(csv_path):
    try:
        df_existing = pd.read_csv(csv_path)
        # Drop duplicates
        df_existing = df_existing.drop_duplicates(subset=["ids"])
        processed_data = dict(zip(df_existing["ids"], df_existing["ocr_text"]))
        print(f"Loaded existing progress. {len(processed_data)} images already processed.")
    except Exception as e:
        print(f"Error loading existing CSV: {e}. Starting fresh.")
        processed_data = {}
else:
    processed_data = {}

images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for image_name in tqdm(images):
    # Check if already processed and has a valid value
    if image_name in processed_data and pd.notna(processed_data[image_name]):
        continue

    image_path = os.path.join(image_folder, image_name)

    try:
        result = ocr.ocr(
            image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=320,
            text_det_limit_type='max'
        )

        text = ""
        if result and result[0]:
            if isinstance(result[0], dict):
                text = " ".join(result[0].get("rec_texts", []))
            else:
                extracted = []
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) > 1 and isinstance(line[1], (list, tuple)) and len(line[1]) > 0:
                        extracted.append(line[1][0])
                text = " ".join(extracted)

    except Exception:
        text = ""

    processed_data[image_name] = text

    # Write incrementally after every image to prevent data loss
    df_temp = pd.DataFrame(list(processed_data.items()), columns=["ids", "ocr_text"])
    df_temp.to_csv(csv_path, index=False)

print(f"Completed! Total processed images saved to {csv_path}.")