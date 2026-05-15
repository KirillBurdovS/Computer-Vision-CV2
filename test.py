import cv2
import numpy as np

image = cv2.imread('/Users/dayeven/Desktop/funnya.jpg', cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

sobel_total = np.sqrt(sobel_x**2 + sobel_y**2)

kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_y = np.array([[-1, -1, -1], [0, 0, 0,], [1, 1, 1]])

prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
prewit_total = np.sqrt(prewitt_x**2 + prewitt_y**2)

_, sobel_edges = cv2.threshold(sobel_total, 100, 255, cv2.THRESH_BINARY)
_, prewit_edges = cv2.threshold(prewit_total, 100, 255, cv2.THRESH_BINARY)

cv2.imwrite('sobel_edges.jpg', sobel_edges.astype(np.uint8))
cv2.imwrite('prewit_edges.jpg', prewit_edges.astype(np.uint8))