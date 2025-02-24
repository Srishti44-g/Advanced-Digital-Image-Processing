import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a pure red image (255, 0, 0)
red_image = np.zeros((300, 300, 3), dtype=np.uint8)
red_image[:, :, 2] = 255  # Set red channel to maximum

# Display the image
plt.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB))
plt.title('Pure Red Image')
plt.axis('off')
plt.show()
