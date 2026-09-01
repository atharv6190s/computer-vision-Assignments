import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, felzenszwalb
from skimage.color import rgb2gray
import os

# 1️⃣ Ensure output folder exists
os.makedirs("Assignment_02", exist_ok=True)

# 2️⃣ Load Image (replace filename with your actual image)
img = cv2.imread('/workspaces/computer-vision-Assignments/Copilot_20260901_194737.png')   
print("Image shape:", img.shape)

# 3️⃣ Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4️⃣ Canny Edge Detection
canny_edges = cv2.Canny(gray, 100, 200)
cv2.imwrite('Assignment_02/canny_output.png', canny_edges)

# 5️⃣ Harris Corner Detection
gray_float = np.float32(gray)
harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
harris = cv2.dilate(harris, None)
img_harris = img.copy()
img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
cv2.imwrite('Assignment_02/harris_output.png', img_harris)

# 6️⃣ Watershed Segmentation
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
img_ws = img.copy()
markers = cv2.watershed(img_ws, markers)
img_ws[markers == -1] = [255,0,0]
cv2.imwrite('Assignment_02/watershed_output.png', img_ws)

# 7️⃣ Felzenszwalb Segmentation
segments_fz = felzenszwalb(rgb2gray(img), scale=100)
plt.imshow(segments_fz, cmap='nipy_spectral')
plt.axis('off')
plt.savefig('Assignment_02/felzenszwalb_output.png', bbox_inches='tight')

print("✅ All outputs saved in Assignment_02 folder: canny_output.png, harris_output.png, watershed_output.png, felzenszwalb_output.png")
