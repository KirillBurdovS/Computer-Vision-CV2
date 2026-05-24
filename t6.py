import cv2 
import torch 
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np 
import easyocr

image_path = '/Users/dayeven/Desktop/banner_s.jpg'

try:
    img_pil = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print("Not fing picture")
    exit()

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.eval()

reader = easyocr.Reader(['ru', 'en'])

inputs = processor(images=img_pil, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# target_sizes = torch.tensor([img_pil.size[::-1]])
h,w = img_pil.size[1], img_pil.size[0]

processed_logits = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])

segmention_mask = processed_logits[0].cpu().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original image")
plt.imshow(img_pil)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("SegFormer")
plt.imshow(segmention_mask, cmap='tab20')
plt.axis("off")

plt.tight_layout()
plt.show()

unique_classes = np.unique(segmention_mask)
print(f"ID find class: {unique_classes}\n")

# Binar mask
binary_mask = (segmention_mask == 43).astype(np.uint8) * 255

contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# For one banner
"""
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    orig_img_np = np.array(img_pil)

    cropped_banner = orig_img_np[y:y+h, x:x+w]
    cropped_banner_bgr = cv2.cvtColor(cropped_banner, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('detected_banner.jpg', cropped_banner)
    print(f"x={x}, y={y}, width={w}, height={h}")

    ocr_results = reader.readtext(cropped_banner, detail=0)

    if len(ocr_results) > 0:
        for i, text in enumerate(ocr_results):
            print(f"Строка {i+1}: {text}")
"""
# More banners
orig_img_np = np.array(img_pil)
if len(contours) > 0: 
    print(f"Всего баннеров: {len(contours)}\n")
    for idx, contours in enumerate(contours):
        if cv2.contourArea(contours) < 100:
            continue

        x,y,w,h = cv2.boundingRect(contours)
        print(f"Баннер: {idx+1}: x={x}, y={y}, width={w}, height={h}")

        cropped_banner = orig_img_np[y:y+h, x:x+w]
        cv2.imwrite(f'detected_banner_{idx+1}.jpg', cv2.cvtColor(cropped_banner, cv2.COLOR_RGB2BGR)) 
        ocr_result = reader.readtext(cropped_banner, detail=0)

        if len(ocr_result) > 0:
            for i, text in enumerate(ocr_result):
                print(f"Строка: {i+1}: {text}")
            else:
                print("invalid text")
            print()
else:
    print("Banner not founded")






