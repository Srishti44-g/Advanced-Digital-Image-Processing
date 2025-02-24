import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('image.jpg', 0)

# Apply Harris Corner Detection
corners = cv2.cornerHarris(image, 2, 3, 0.04)

# Mark the corners in the image
image[corners > 0.01 * corners.max()] = [0, 0, 255]  # Red color for corners

# Display the image with corners
plt.imshow(image)
plt.title('Corners Detected')
plt.axis('off')
plt.show()
