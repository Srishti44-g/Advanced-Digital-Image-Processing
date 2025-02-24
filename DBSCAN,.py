import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to RGB (from BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
image_reshaped = image_rgb.reshape((-1, 3))

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=30, min_samples=100)  # eps controls the neighborhood radius
dbscan.fit(image_reshaped)

# Get the cluster labels
labels = dbscan.labels_

# Assign different colors to each cluster
unique_labels = np.unique(labels)
segmented_image = np.zeros_like(image_rgb)

for label in unique_labels:
    cluster_pixels = image_reshaped[labels == label]
    # Assign a random color to each cluster
    random_color = np.random.randint(0, 256, 3)
    segmented_image[labels == label] = random_color

# Plot the original and segmented images
fig, ax = plt.subplots(1, 2, figsize=(15, 10))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Image Segmentation (DBSCAN)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
