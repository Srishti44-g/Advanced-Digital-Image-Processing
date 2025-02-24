# Convert from RGB to HSV
image_rgb = cv2.imread('image.jpg')
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

# Display the image in HSV color space
plt.imshow(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV Image')
plt.axis('off')
plt.show()
