import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg', 0)  # Load image in grayscale

# 1. Histogram Equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# 2. Contrast Stretching
def contrast_stretching(image, min_out=0, max_out=255):
    min_in = np.min(image)
    max_in = np.max(image)
    stretched_image = (image - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
    stretched_image = np.clip(stretched_image, 0, 255)
    return np.uint8(stretched_image)

# 3. Gamma Correction
def gamma_correction(image, gamma=1.0):
    # Normalize the image to [0, 1]
    image_normalized = image / 255.0
    corrected_image = np.power(image_normalized, gamma) * 255
    corrected_image = np.clip(corrected_image, 0, 255)
    return np.uint8(corrected_image)

# Apply Histogram Equalization
equalized_image = histogram_equalization(image)

# Apply Contrast Stretching
stretched_image = contrast_stretching(image)

# Apply Gamma Correction (example gamma = 0.5 for brightening)
gamma_corrected_image = gamma_correction(image, gamma=0.5)

# Plotting the images
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Original Image
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

# Histogram Equalized Image
ax[0, 1].imshow(equalized_image, cmap='gray')
ax[0, 1].set_title('Histogram Equalized Image')
ax[0, 1].axis('off')

# Contrast Stretched Image
ax[0, 2].imshow(stretched_image, cmap='gray')
ax[0, 2].set_title('Contrast Stretched Image')
ax[0, 2].axis('off')

# Gamma Corrected Image
ax[1, 0].imshow(gamma_corrected_image, cmap='gray')
ax[1, 0].set_title('Gamma Corrected Image (Gamma=0.5)')
ax[1, 0].axis('off')

# Histogram of the original image
ax[1, 1].hist(image.ravel(), bins=256, color='gray', histtype='step')
ax[1, 1].set_title('Histogram of Original Image')

# Histogram of the equalized image
ax[1, 2].hist(equalized_image.ravel(), bins=256, color='gray', histtype='step')
ax[1, 2].set_title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()
