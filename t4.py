import cv2
import torch 
import torch.nn.functional as F 
import numpy as np

img = cv2.imread('/Users/dayeven/Desktop/wis.jpg', cv2.IMREAD_GRAYSCALE)
img_small = cv2.resize(img, (400, 400)) 

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [1, 0, 1]
])

img_tensor = torch.from_numpy(img_small).float()[None, None, :, :]
kernel_tensor = torch.from_numpy(sobel_x).float()[None, None, :, :]

result = F.conv2d(img_tensor, kernel_tensor, stride=1, padding=0)

fast_edges_x = result.squeeze().numpy() 
fast_edges_x = np.abs(fast_edges_x)

if fast_edges_x.max() > 0:
    fast_edges_x = (fast_edges_x / fast_edges_x.max() * 255).astype(np.uint8)
else:
    fast_edges_x = fast_edges_x.astype(np.uint8)

cv2.imwrite('fast_edges_sobel.jpg', fast_edges_x)
