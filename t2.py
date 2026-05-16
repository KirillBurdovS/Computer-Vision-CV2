import numpy as np 

def manual_conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape 

    out_h = int((img_h-kernel_h)/ stride) + 1
    out_w = int((img_w-kernel_w)/ stride) + 1

    output = np.zeros((out_h, out_w))

    for y in range(out_h):
        for x in range(out_w):
            y_start = y * stride
            y_end = y_start + kernel_h

            x_start = x * stride 
            x_end = x_start + kernel_w

            image_path = image[y_start:y_end, x_start:x_end]
            output[y, x] = np.sum(image_path * kernel)

    return output

test_image = np.array([
    [0, 0, 10, 10, 10],
    [0, 0, 10, 10, 10],
    [0, 0, 10, 10, 10],
    [0, 0, 10, 10, 10],
    [0, 0, 10, 10, 10]
])

sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

result = manual_conv2d(test_image, sobel_vertical, stride=1, padding=0)
print(test_image)
print("Результат:")
print(result)




