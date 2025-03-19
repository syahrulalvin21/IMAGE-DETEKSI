# IMAGE-DETEKSI

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = "walpaper.jpeg"  # Sesuaikan dengan path gambar Anda
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cek apakah gambar berhasil dimuat
if image is None:
    raise ValueError("Gagal memuat gambar. Periksa path gambar.")

# 1. Citra Negatif
negative = 255 - image

# 2. Transformasi Log
c = 255 / (np.log(1 + np.max(image) + 1e-6))  # Hindari log(0)
log_transform = c * np.log(1 + image.astype(np.float32))
log_transform = np.clip(log_transform, 0, 255).astype(np.uint8)

# 3. Transformasi Power Law (Gamma Correction)
gamma = 2.2  # Nilai gamma bisa disesuaikan
power_law = np.clip(255 * ((image / 255 + 1e-6) ** gamma), 0, 255).astype(np.uint8)

# 4. Histogram Equalization
hist_eq = cv2.equalizeHist(image)

# 5. Histogram Normalization
norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# 6. Konversi RGB ke HSI (opsional)
def rgb_to_hsi(image):
    if image is None:
        raise ValueError("Gagal memuat gambar berwarna.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255
    R, G, B = cv2.split(image)
    
    # H, S, I initialization
    I = (R + G + B) / 3
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB
    
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))
    
    H = np.where(B > G, 2 * np.pi - theta, theta)
    H = (H / (2 * np.pi)) * 255
    
    return cv2.merge([H, S * 255, I * 255]).astype(np.uint8)

image_rgb = cv2.imread(image_path)
hsi_image = rgb_to_hsi(image_rgb)

# Menampilkan hasil
titles = ['Original', 'Negative', 'Log Transform', 'Power Law', 
          'Histogram Equalization', 'Histogram Normalization']
images = [image, negative, log_transform, power_law, hist_eq, norm_image]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
