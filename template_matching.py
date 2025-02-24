import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the main image and template image
image = cv2.imread('image.jpg')
template = cv2.imread('template.jpg')

# Convert the images to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

# Get the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Get the dimensions of the template
w, h = gray_template.shape[::-1]

# Draw a rectangle around the matched region
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Convert the image from BGR to RGB for plotting
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plotting the result
plt.imshow(image_rgb)
plt.title('Template Matching (Object Detection)')
plt.axis('off')
plt.show()
