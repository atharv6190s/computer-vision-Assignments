import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, felzenszwalb
from skimage.color import rgb2gray
from skimage.feature import hog
import os

# Ensure output folder exists
os.makedirs("Assignment_02", exist_ok=True)

# Load Image (replace filename with your actual image)
img = cv2.imread("Copilot_20260901_194737.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------ Task 2: Edge + Corner Detection ------------------
canny = cv2.Canny(gray, 100, 200)
cv2.imwrite("Assignment_02/canny_output.png", canny)

harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("Assignment_02/harris_output.png", harris_norm)

# ------------------ Task 3: Segmentation ------------------
gray_float = rgb2gray(img)
watershed_seg = watershed(gray_float)
felzen_seg = felzenszwalb(gray_float, scale=100)

plt.imsave("Assignment_02/watershed_output.png", watershed_seg, cmap="nipy_spectral")
plt.imsave("Assignment_02/felzenszwalb_output.png", felzen_seg, cmap="nipy_spectral")

# ------------------ Task 4: SIFT + HOG Descriptors ------------------
# Convert to grayscale for SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

sift_img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("Assignment_02/sift_output.png", sift_img)

# HOG descriptor
hog_features, hog_image = hog(gray, visualize=True, block_norm='L2-Hys')
plt.imsave("Assignment_02/hog_output.png", hog_image, cmap="gray")

# ------------------ Task 5: Morphological Operations ------------------
# Binary mask from Canny for demo
_, mask = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(mask, kernel, iterations=1)
dilation = cv2.dilate(mask, kernel, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("Assignment_02/morph_erosion.png", erosion)
cv2.imwrite("Assignment_02/morph_dilation.png", dilation)
cv2.imwrite("Assignment_02/morph_opening.png", opening)
cv2.imwrite("Assignment_02/morph_closing.png", closing)

print("✅ Assignment 2 complete: Outputs saved in Assignment_02 folder")
