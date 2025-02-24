import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('image.jpg', 0)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Display the original and edge-detected images
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

plt.show()
