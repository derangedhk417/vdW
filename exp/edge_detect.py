import sys
import cv2
import code
import numpy as np
from matplotlib import pyplot as plt

def getLargestContour(contours):
	max_size = 0.0
	largest  = None

	for c in contours:
		area = cv2.contourArea(c)
		if area > max_size:
			max_size = area
			largest  = c

	return c, max_size

def getLargerThan(contours, s):
	selected = []

	for c in contours:
		area = cv2.contourArea(c)
		if area > s:
			selected.append(c)

	return selected

# Load the image in greyscale
img   = cv2.imread('img.png', cv2.IMREAD_COLOR)

# Detect the edges
edges = cv2.Canny(img, 10, 75)

# Extract contour vectors from the edges.
contours, heirarchy = cv2.findContours(
	edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# code.interact(local=locals())

large_ones = getLargerThan(contours, 50.0)

# Draw the largest contours.
big = img.copy()
for i in range(len(large_ones)):
	big = cv2.drawContours(big, large_ones, i, (255, 0, 0), 1)

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(big, cmap='gray')
plt.title('Contours Larger than 25.0')
plt.xticks([])
plt.yticks([])

plt.show()