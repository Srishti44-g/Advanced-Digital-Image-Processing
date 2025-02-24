import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use the Canny edge detector
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edge-detected image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contoured_image = image.copy()
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)

# Convert the image from BGR to RGB for plotting
contoured_image_rgb = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB)

# Plotting the result
plt.imshow(contoured_image_rgb)
plt.title('Object Detection (Contours)')
plt.axis('off')
plt.show()
