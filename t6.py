import torch 
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np 

image_path = '/Users/dayeven/Desktop/banner_s.jpg'

try:
    img_pil = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print("Not fing picture")
    exit()


processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

model.eval()

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


