from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="te")

image = r"C:\Users\Dell\Documents\project\research_project\dataset\images\image_telugu_0001.png"

result = ocr.predict(image)

print(result)