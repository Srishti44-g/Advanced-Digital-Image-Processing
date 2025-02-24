import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to RGB (from BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
image_reshaped = image_rgb.reshape((-1, 3))

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters (k)
kmeans.fit(image_reshaped)

# Get the cluster centers (the mean color for each cluster)
cluster_centers = kmeans.cluster_centers_

# Assign each pixel to the nearest cluster
segmented_image = cluster_centers[kmeans.labels_]
segmented_image = segmented_image.reshape(image_rgb.shape)

# Convert the segmented image back to the original range (0-255)
segmented_image = np.uint8(segmented_image)

# Plot the original and segmented images
fig, ax = plt.subplots(1, 2, figsize=(15, 10))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Image Segmentation (K-means)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
