import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg', 0)  # Read in grayscale

# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Canny Edge Detection
edges = cv2.Canny(blurred_image, 100, 200)

# Plotting the original image and edge-detected image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(edges, cmap='gray')
ax[1].set_title('Edge Detected Image (Canny)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
