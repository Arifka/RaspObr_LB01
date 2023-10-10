import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
import plot_creator
import correction

img_name = '2'
img = cv.imread(img_name + '.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
# Task 1
plt.figure("Histograms of original brightness")
plt.subplot(211)
plt = plot_creator.create_plot(0, img)

grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(212)
plt = plot_creator.create_plot(1, grayscale_img)

cv.imwrite(img_name + '_gray' + '.jpg', grayscale_img)
print("End of Task 1")

# Task 2. Color
plt.figure("Historgams of liner correction")
# Compressed
new_img_compressed = correction.compression_image(0, img)
# Save compressed img
cv.imwrite(img_name + '_compressed' + '.jpg', new_img_compressed)
# Linier Correction
new_img_linCorr = correction.linier_correction(0, new_img_compressed)
# Save img
cv.imwrite(img_name + '_linCorr' + '.jpg', new_img_linCorr)
# Histograms of brightness
plt.subplot(211)
plt = plot_creator.create_plot(0, new_img_linCorr)

#Task 2 GrayScale
# Compressed
new_grayscale_img_compressed = correction.compression_image(1, grayscale_img)
# Save compressed grayscale_img
cv.imwrite(img_name + '_gray' + '_compressed' + '.jpg', new_grayscale_img_compressed)
# Change brightness with formule
new_grayscale_img_linCorr = correction.linier_correction(1, new_grayscale_img_compressed)
# Save result
cv.imwrite(img_name + '_gray' + '_linCorr' + '.jpg', new_grayscale_img_linCorr)
# Histogram of brightness
plt.subplot(212)
plt = plot_creator.create_plot(1, new_grayscale_img_linCorr)
print("End of Task 2")

# Task 3.1 GammaCor
# Colorized
plt.figure("Histogram of britghtness after gamma correction")
new_img_gammaCorr = correction.gamma_correction(0, new_img_compressed)
cv.imwrite(img_name + '_gammaCorr' + '.jpg', new_img_gammaCorr)
# Histogram of brightness
plt.subplot(211)
plt = plot_creator.create_plot(0, new_img_gammaCorr)

# Grayscale
new_grayscale_img_gammaCorr = correction.gamma_correction(1, new_grayscale_img_compressed)
cv.imwrite(img_name + '_gray' + '_gammaCorr' + '.jpg', new_grayscale_img_gammaCorr)
# Histogram of brightness
plt.subplot(212)
plt = plot_creator.create_plot(1, new_grayscale_img_gammaCorr)

# Task 3.2 Logarithm
# Colorized
new_img_logCorr = correction.logarithm_correction(0, new_img_compressed)
cv.imwrite(img_name + '_logCorr' + '.jpg', new_img_logCorr)
# Histograms of brightness
plt.figure("Histograms of brightness after logarithm correction")
plt.subplot(211)
plt = plot_creator.create_plot(0, new_img_logCorr)
# Grayscale
new_grayscale_img_logCorr = correction.logarithm_correction(1, new_grayscale_img_compressed)
cv.imwrite(img_name + '_gray' + '_logCorr' + '.jpg', new_grayscale_img_logCorr)
# Histograms
plt.subplot(212)
plt = plot_creator.create_plot(1, new_grayscale_img_logCorr)

plt.tight_layout()
plt.show()
print("End of Task 3")