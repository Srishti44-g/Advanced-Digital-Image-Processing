import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg', 0)

# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Use HoughCircles to detect circles
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=15, maxRadius=50)

# Convert circles to integer
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # Draw circles on the original image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Convert the image from BGR to RGB for plotting
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plotting the result
plt.imshow(image_rgb)
plt.title('Circle Detection (Hough Transform)')
plt.axis('off')
plt.show()
