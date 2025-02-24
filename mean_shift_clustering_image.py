import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to RGB (from BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
image_reshaped = image_rgb.reshape((-1, 3))

# Apply MeanShift clustering
mean_shift = MeanShift(bandwidth=30)  # Define bandwidth for the window size
mean_shift.fit(image_reshaped)

# Get the labels (cluster assignments)
segmented_image = mean_shift.cluster_centers_[mean_shift.labels_]
segmented_image = segmented_image.reshape(image_rgb.shape)

# Convert the segmented image back to the original range (0-255)
segmented_image = np.uint8(segmented_image)

# Plot the original and segmented images
fig, ax = plt.subplots(1, 2, figsize=(15, 10))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Image Segmentation (Mean Shift)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
