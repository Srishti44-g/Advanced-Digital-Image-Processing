import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

# 1. Scaling
scale_factor = 0.5
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
dim = (width, height)
scaled_image = cv2.resize(image, dim)

# 2. Rotation
angle = 45  # Rotation angle in degrees
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# 3. Translation
tx, ty = 100, 50  # Translate by 100 pixels horizontally and 50 pixels vertically
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# 4. Flipping (Horizontally)
flipped_image = cv2.flip(image, 1)  # 1 means flip horizontally

# Plotting all transformed images
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0, 0].imshow(image)
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

ax[0, 1].imshow(scaled_image)
ax[0, 1].set_title('Scaled Image')
ax[0, 1].axis('off')

ax[0, 2].imshow(rotated_image)
ax[0, 2].set_title('Rotated Image')
ax[0, 2].axis('off')

ax[1, 0].imshow(translated_image)
ax[1, 0].set_title('Translated Image')
ax[1, 0].axis('off')

ax[1, 1].imshow(flipped_image)
ax[1, 1].set_title('Flipped Image')
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()
