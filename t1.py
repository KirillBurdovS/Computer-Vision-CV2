import cv2 
import numpy as np 

image = cv2.imread('/Users/dayeven/Desktop/funnya.jpg')

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Тон, Насыщенность, Яркость
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h_schannel, s_schannel, v_schannel = cv2.split(img_hsv)
cv2.imwrite('hsv_v_brightness.jpg', v_schannel.astype(np.uint8)) # astype(np.uint8) для 8 Bit 

img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
cv2.imwrite('ycrcb_y_luminance.jpg', y_channel.astype(np.uint8))

# Края 
b_ch, g_ch, r_ch = cv2.split(image)

def apply_sobel(channel):
    sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)

def apply_prewitt(channel):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    prewitt_x = cv2.filter2D(channel, cv2.CV_64F, kx)
    prewitt_y = cv2.filter2D(channel, cv2.CV_64F, ky)
    return np.sqrt(prewitt_x**2 + prewitt_y**2)

# Собель для кажждого канала отдельно
sobel_b = apply_sobel(b_ch)
sobel_g = apply_sobel(g_ch)
sobel_r = apply_sobel(r_ch)

color_sobel_total = np.maximum(np.maximum(sobel_b, sobel_g), sobel_r)

prewitt_b = apply_prewitt(b_ch)
prewitt_g = apply_prewitt(g_ch)
prewitt_r = apply_prewitt(r_ch)
color_prewitt_total = np.maximum(np.maximum(prewitt_b, prewitt_g), prewitt_r)

THRESHOLD_VALUE = 80

_, edges_sobel = cv2.threshold(color_sobel_total, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
_, edges_prewitt = cv2.threshold(color_prewitt_total, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

cv2.imwrite('color_edges_sobel.jpg', edges_sobel.astype(np.uint8))
cv2.imwrite('prewitt_edges_color.jpg', edges_prewitt.astype(np.uint8))

