import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- LOAD IMAGE ----------
img = cv2.imread("data/download.jpg")
if img is None:
    raise ValueError("Image not found. Check the path.")

# ---------- GRAYSCALE ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- CLAHE ----------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
contrast_enhanced = clahe.apply(gray)

# ---------- NOISE REMOVAL ----------
blur = cv2.GaussianBlur(contrast_enhanced, (3,3), 0)

# ---------- THRESHOLDING ----------
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert so platelets are white
binary = cv2.bitwise_not(binary)

# ---------- REMOVE SMALL NOISE ----------
# Smaller kernel and fewer iterations to preserve tiny platelets
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# ---------- SURE BACKGROUND AND FOREGROUND ----------
# Background (sure area) via dilation
sure_bg = cv2.dilate(opening, kernel, iterations=1)

# Distance transform for foreground (sure platelets)
# Smaller mask for better small-platelet sensitivity
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

# Lower threshold for foreground to include tiny platelets
_, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)

# Optional: dilate foreground slightly to strengthen tiny platelet markers
sure_fg = cv2.dilate(np.uint8(sure_fg), np.ones((2,2), np.uint8), iterations=1)

# Convert foreground to uint8
sure_fg = np.uint8(sure_fg)

# Unknown region (boundary between objects)
unknown = cv2.subtract(sure_bg, sure_fg)

# ---------- MARKERS ----------
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so background != 0
markers = markers + 1

# Mark unknown region with 0
markers[unknown==255] = 0

# ---------- APPLY WATERSHED ----------
markers = cv2.watershed(img, markers)

# Boundary marking: pixels marked -1 are boundaries
watershed_img = img.copy()
watershed_img[markers == -1] = [0, 0, 255]  # red boundaries

# ---------- DISPLAY ----------
plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(binary, cmap='gray')
plt.title("Binary after CLAHE & Otsu")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(dist_transform, cmap='jet')
plt.title("Distance Transform")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB))
plt.title("Watershed Segmentation")
plt.axis("off")


plt.tight_layout()
plt.show()
