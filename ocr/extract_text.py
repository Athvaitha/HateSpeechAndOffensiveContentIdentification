import os
import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

# Initialize OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="te"     # Telugu
)

image_folder = r"C:\Users\Dell\Documents\project\research_project\Telugu_HASOC\Telugu_HASOC\images_all"

data = []

images = sorted(os.listdir(image_folder))

for image_name in tqdm(images):

    image_path = os.path.join(image_folder, image_name)

    try:
        result = ocr.ocr(image_path)

        extracted = []

        if result and result[0]:

            for line in result[0]:
                extracted.append(line[1][0])

        text = " ".join(extracted)

    except Exception:

        text = ""

    data.append([image_name, text])

df = pd.DataFrame(data, columns=["ids", "ocr_text"])

df.to_csv("dataset/ocr_text.csv", index=False)

print(df.head())