from skimage.feature import hog
from skimage import data, color, exposure
import matplotlib.pyplot as plt

# Load sample image
image = data.astronaut()

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Compute HOG features and the corresponding image representation
features, hog_image = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

# Rescale the HOG image for better visibility
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Plotting the result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray_image, cmap=plt.cm.gray)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax[1].set_title('HOG Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()
