import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load Images (adjust your paths here)
# -------------------------------
img_normal = cv2.imread("/workspaces/computer-vision/first image (1) CV ASSIGNMENT.jpeg", cv2.IMREAD_GRAYSCALE)
img_under  = cv2.imread("/workspaces/computer-vision/Second image (2) CV ASSIGNMENT.jpeg", cv2.IMREAD_GRAYSCALE)
img_over   = cv2.imread("/workspaces/computer-vision/Third image (3) CV ASSIGNMENT.jpeg", cv2.IMREAD_GRAYSCALE)

# -------------------------------
# Q1: Radiometry & Digitization (Histograms)
# -------------------------------
def show_histogram(image, title, filename):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=[0,256])   # FIXED: no warning
    plt.title(title)
    plt.savefig(filename)
    plt.close()

show_histogram(img_normal, "Normal Image Histogram", "hist_normal.png")
show_histogram(img_under, "Under-exposed Histogram", "hist_under.png")
show_histogram(img_over, "Over-exposed Histogram", "hist_over.png")

# -------------------------------
# Q2: Custom Convolution (Blur & Sharpen)
# -------------------------------
def custom_convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i,j] = np.sum(region * kernel)
    return output

# Blur kernel
blur_kernel = np.ones((3,3), np.float32)/9
blurred_custom = custom_convolution(img_normal, blur_kernel)
cv2.imwrite("blurred_custom.png", np.uint8(blurred_custom))

# Sharpen kernel
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpened_custom = custom_convolution(img_normal, sharpen_kernel)
cv2.imwrite("sharpened_custom.png", np.uint8(sharpened_custom))

# -------------------------------
# Q3: Histogram Equalization
# -------------------------------
equalized = cv2.equalizeHist(img_under)
cv2.imwrite("equalized_under.png", np.uint8(equalized))

# -------------------------------
# Q4: Fourier Transform & Frequency Filtering
# -------------------------------
f = np.fft.fft2(img_normal)
fshift = np.fft.fftshift(f)
magnitude = 20*np.log(np.abs(fshift))

plt.figure()
plt.imshow(magnitude, cmap='gray')
plt.title("Magnitude Spectrum")
plt.savefig("magnitude_spectrum.png")
plt.close()

# Create mask to remove high-frequency noise
rows, cols = img_normal.shape
crow, ccol = rows//2 , cols//2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)

cv2.imwrite("filtered_image.png", np.uint8(img_filtered))

# -------------------------------
# Q5: Transformations (Rigid vs Affine)
# -------------------------------
rows, cols = img_normal.shape

# Rigid Transformation (rotation + translation)
M_rigid = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)  # rotate 30°
rigid = cv2.warpAffine(img_normal, M_rigid, (cols, rows))
cv2.imwrite("rigid_transform.png", np.uint8(rigid))

# Affine Transformation (rotation + shearing)
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img_normal, M_affine, (cols, rows))
cv2.imwrite("affine_transform.png", np.uint8(affine))
