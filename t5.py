import cv2 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn.functional as F 

img = cv2.imread('/Users/dayeven/Desktop/zak.jpg', cv2.IMREAD_GRAYSCALE)

img_small = cv2.resize(img, (400 ,400))

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0 , 0],
    [1, 2, 1]
]) 

img_tensor = torch.from_numpy(img_small).float()[None, None, :, :]
kernel_x_tensor = torch.from_numpy(sobel_x).float()[None, None, :, :]
kernel_y_tensor = torch.from_numpy(sobel_y).float()[None, None, :, :]

Gx = F.conv2d(img_tensor, kernel_x_tensor, stride=1, padding=0).squeeze().numpy()
Gy = F.conv2d(img_tensor, kernel_y_tensor, stride=1, padding=0).squeeze().numpy()

G_total = np.sqrt(Gx**2 + Gy**2)

if G_total.max() > 0:
    G_total = (G_total / G_total.max() * 255).astype(np.uint8)
else:
    G_total = G_total.astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Горизонтальный (Gx)")
plt.imshow(np.abs(Gx), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Вертикальный (Gy)")
plt.imshow(np.abs(Gy), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Полный вектор градиента (G)")
plt.imshow(np.abs(G_total), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()