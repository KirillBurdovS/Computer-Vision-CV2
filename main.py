import numpy as np

I = np.array([
    [60, 70, 80],
    [160, 170, 180],
    [250, 255, 255]
])

kx_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
ky_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

Gx_sobel = np.sum(I * kx_sobel)
Gy_sobel = np.sum(I * ky_sobel)
G_sobel = np.sqrt(Gx_sobel**2 + Gy_sobel**2)

print(f"Gx: {Gx_sobel}, Gy: {Gy_sobel}")
print(f"Градиент G: {G_sobel:.2f}\n")

Kx_previt = np.array([[-1, 0, 1], [-1, 0, 1], [1, 0, 1]])
Ky_previt = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

Gx_previt = np.sum(I * Kx_previt)
Gy_previt = np.sum(I * Ky_previt)
G_previt = np.sqrt(Gx_previt**2 + Gy_previt**2)

print(f"Gx: {Gx_previt}, Gy: {Gy_previt}")
print(f"Градиент: {G_previt:.2f}\n")
