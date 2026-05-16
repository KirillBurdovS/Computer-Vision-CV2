import cv2
import numpy as np
import torch 
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import requests
from io import BytesIO

object1 = cv2.imread("/Users/dayeven/Desktop/marat.jpg")
object_bgr= cv2.imread("/Users/dayeven/Desktop/zak.jpg")

bg_resized = cv2.resize(object_bgr, (512, 512))
object_resized = cv2.resize(object1, (200, 200))

bg_resized[300:500, 300:500] = object_resized

collage_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
init_image = Image.fromarray(collage_rgb)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

if torch.backends.mps.is_available():
    pipe = pipe.to("mps")
else:
    pipe = pipe.to("cpu")

pipe.enable_attention_slicing()

promt = "A high-quality photo of a man standing in the foreground, beautiful cinematic background, photorealistic, highly detailed, professional photography, 4k, realistic lighting and soft shadows"
images = pipe(
    prompt=promt,
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=30
)

final_image = images.images[0]
final_image.save('natural_picture.jpg')
final_image.show()
