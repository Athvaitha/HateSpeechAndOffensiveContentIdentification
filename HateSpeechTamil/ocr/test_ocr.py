import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="ta", enable_mkldnn=False)

image = r"dataset/images/image_tamil_0001.jpg"

result = ocr.ocr(image)

print(result)