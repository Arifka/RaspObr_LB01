import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt

img_name = '2'
img = cv.imread(img_name + '.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
# Task 1
plt.figure("Histograms of original brightness")
plt.subplot(211)
color = ('b','g','r')
for i,col in enumerate(color):
   histr = cv.calcHist([img],[i],None,[256],[0,256])
   plt.plot(histr,color = col)
   plt.xlim([0,256])

plt.subplot(212)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
histr = cv.calcHist([grayscale_img], [0], None, [256], [0,256])
plt.plot(histr)
plt.xlim([0, 256])
cv.imwrite(img_name + '_gray' + '.jpg', grayscale_img)
print("End of Task 1")

# Task 2. Color
plt.figure("Historgams of liner correction")
new_img_compressed = np.zeros(img.shape, img.dtype)
alpha = 1.0
beta = 0.0
# Compressed brightness in [50, 200]
for y in range(img.shape[0]):
   for x in range(img.shape[1]):
      for channel in range(img.shape[2]):
         new_img_compressed[y, x, channel] = np.clip(alpha*img[y, x, channel]+beta, 50, 200)
cv.imwrite(img_name + '_compressed' + '.jpg', new_img_compressed)

# Linier Correction
b_min = new_img_compressed[0, 0, 0]
r_min = new_img_compressed[0, 0, 1]
g_min = new_img_compressed[0, 0, 2]
list_min = [b_min, r_min, g_min]

b_max = new_img_compressed[0, 0, 0]
r_max = new_img_compressed[0, 0, 1]
g_max = new_img_compressed[0, 0, 2]
list_max = [b_max, r_max, g_max]
#Find min, max of BRG pixel brightness
for y in range(new_img_compressed.shape[0]):
   for x in range(new_img_compressed.shape[1]):
      for c in range(new_img_compressed.shape[2]):
         list_min[c] = min(list_min[c], new_img_compressed[y, x, c])
         list_max[c] = max(list_max[c], new_img_compressed[y, x, c])

# Change brightness with formule
new_img_linCorr = np.zeros(new_img_compressed.shape, new_img_compressed.dtype)
for y in range(new_img_compressed.shape[0]):
   for x in range(new_img_compressed.shape[1]):
      for channel in range(new_img_compressed.shape[2]):
         new_img_linCorr[y, x, channel] = (new_img_compressed[y, x, channel] - list_min[channel]) * 255 / (list_max[channel] - list_min[channel])

# Save img
cv.imwrite(img_name + '_linCorr' + '.jpg', new_img_linCorr)
# Histograms of brightness
plt.subplot(211)
for i,col in enumerate(color):
   histr = cv.calcHist([new_img_linCorr],[i],None,[256],[0,256])
   plt.plot(histr,color = col)
   plt.xlim([0,256])

#Task 2 GrayScale
# Compressed
new_grayscale_img_compressed = np.zeros(grayscale_img.shape, grayscale_img.dtype)
for y in range(grayscale_img.shape[0]):
   for x in range(grayscale_img.shape[1]):
      new_grayscale_img_compressed[y, x] = np.clip(alpha*grayscale_img[y, x]+beta, 50, 200)
# Save compressed grayscale_img
cv.imwrite(img_name + '_gray' + '_compressed' + '.jpg', new_grayscale_img_compressed)
# Find min, max
brightness_min = new_grayscale_img_compressed[0, 0]
brightness_max = new_grayscale_img_compressed[0, 0]
for y in range(new_grayscale_img_compressed.shape[0]):
   for x in range(new_grayscale_img_compressed.shape[1]):
      brightness_min = min(brightness_min, new_grayscale_img_compressed[y, x])
      brightness_max = max(brightness_max, new_grayscale_img_compressed[y, x])
# Change brightness with formule
new_grayscale_img_linCorr = np.zeros(new_grayscale_img_compressed.shape, new_grayscale_img_compressed.dtype)
for y in range(new_grayscale_img_compressed.shape[0]):
   for x in range(new_grayscale_img_compressed.shape[1]):
      new_grayscale_img_linCorr[y, x] = (new_grayscale_img_compressed[y, x] - brightness_min)*255/(brightness_max - brightness_min)
# Save result
cv.imwrite(img_name + '_gray' + '_linCorr' + '.jpg', new_grayscale_img_linCorr)
# Histogram of brightness
plt.subplot(212)
histr = cv.calcHist([new_grayscale_img_linCorr], [0], None, [256], [0,256])
plt.plot(histr)
plt.xlim([0, 256])
print("End of Task 2")

# Task 3.1 GammaCor
# Colorized
plt.figure("Histogram of britghtness after gamma correction")
new_img_gammaCorr = np.zeros(new_img_compressed.shape, new_img_compressed.dtype)
coef_c = 1.55
coef_gamma = 2
for y in range(new_img_compressed.shape[0]):
   for x in range(new_img_compressed.shape[1]):
      for c in range(new_img_compressed.shape[2]):
         new_img_gammaCorr[y, x, c] = 255 * coef_c * (new_img_compressed[y, x, c]/255) ** coef_gamma
# Histogram of brightness
plt.subplot(211)
for i,col in enumerate(color):
   histr = cv.calcHist([new_img_gammaCorr],[i],None,[256],[0,256])
   plt.plot(histr,color = col)
   plt.xlim([0,256])

cv.imwrite(img_name + '_gammaCorr' + '.jpg', new_img_gammaCorr)
# Grayscale
new_grayscale_img_gammaCorr = np.zeros(new_grayscale_img_compressed.shape, new_grayscale_img_compressed.dtype)
for y in range(new_grayscale_img_compressed.shape[0]):
   for x in range(new_grayscale_img_compressed.shape[1]):
      new_grayscale_img_gammaCorr[y, x] = 255 * coef_c * (new_grayscale_img_compressed[y, x] / 255) ** coef_gamma
cv.imwrite(img_name + '_gray' + '_gammaCorr' + '.jpg', new_grayscale_img_gammaCorr)
# Histogram of brightness
plt.subplot(212)
histr = cv.calcHist([new_grayscale_img_gammaCorr], [0], None, [256], [0,256])
plt.plot(histr)
plt.xlim([0, 256])
# Task 3.2 Logarithm
# Colorized
new_img_logCorr = np.zeros(new_img_compressed.shape, new_img_compressed.dtype)
for y in range(new_img_compressed.shape[0]):
   for x in range(new_img_compressed.shape[1]):
      for c in range(new_img_compressed.shape[2]):
         new_img_logCorr[y, x, c] = 255 * coef_c * math.log((new_img_compressed[y, x, c] / 255) + 1)
# Histograms of brightness
plt.figure("Histograms of brightness after logarithm correction")
plt.subplot(211)
for i,col in enumerate(color):
   histr = cv.calcHist([new_img_logCorr],[i],None,[256],[0,256])
   plt.plot(histr,color = col)
   plt.xlim([0,256])
cv.imwrite(img_name + '_logCorr' + '.jpg', new_img_logCorr)
# Grayscale
new_grayscale_img_logCorr = np.zeros(new_grayscale_img_compressed.shape, new_grayscale_img_compressed.dtype)
for y in range(new_grayscale_img_compressed.shape[0]):
   for x in range(new_grayscale_img_compressed.shape[1]):
      new_grayscale_img_logCorr[y, x] = 255 * coef_c * math.log((new_grayscale_img_compressed[y, x] / 255) + 1)
# Histograms
plt.subplot(212)
histr = cv.calcHist([new_grayscale_img_logCorr], [0], None, [256], [0,256])
plt.plot(histr)
plt.xlim([0, 256])
cv.imwrite(img_name + '_gray' + '_logCorr' + '.jpg', new_grayscale_img_logCorr)

plt.tight_layout()
plt.show()
print("End of Task 3")