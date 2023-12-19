import math
import cv2 as cv
from matplotlib import pyplot as plt
import plot_creator
import correction


img_name = "2"

# Task 1
img = cv.imread(img_name + ".jpg")
assert img is not None, "file could not be read, check with os.path.exists()"
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite(img_name + "_gray" + ".jpg", grayscale_img)
plt = plot_creator.get_plot("Histograms of original brightness", img, grayscale_img)
print("End of Task 1")

# Task 2. Color
# Compressed
new_img_compressed = correction.compression_image(False, img)
cv.imwrite(img_name + "_compressed" + ".jpg", new_img_compressed)
# Linier Correction
new_img_linCorr = correction.linier_correction(False, new_img_compressed)
cv.imwrite(img_name + "_linCorr" + ".jpg", new_img_linCorr)
# GrayScale
# Compressed
new_grayscale_img_compressed = correction.compression_image(True, grayscale_img)
cv.imwrite(img_name + "_gray" + "_compressed" + ".jpg", new_grayscale_img_compressed)
# Linier Correction
new_grayscale_img_linCorr = correction.linier_correction(
    True, new_grayscale_img_compressed
)
cv.imwrite(img_name + "_gray" + "_linCorr" + ".jpg", new_grayscale_img_linCorr)
# Histogram of brightness
plt = plot_creator.get_plot(
    "Historgams of liner correction", new_img_linCorr, new_grayscale_img_linCorr
)
print("End of Task 2")

# Task 3.1 GammaCor
# Colorized
new_img_gammaCorr = correction.gamma_correction(False, new_img_compressed)
cv.imwrite(img_name + "_gammaCorr" + ".jpg", new_img_gammaCorr)
# Grayscale
new_grayscale_img_gammaCorr = correction.gamma_correction(
    True, new_grayscale_img_compressed
)
cv.imwrite(img_name + "_gray" + "_gammaCorr" + ".jpg", new_grayscale_img_gammaCorr)
# Histogram of brightness
plt = plot_creator.get_plot(
    "Histogram of britghtness after gamma correction",
    new_img_gammaCorr,
    new_grayscale_img_gammaCorr,
)

# Task 3.2 Logarithm
# Colorized
new_img_logCorr = correction.logarithm_correction(False, new_img_compressed)
cv.imwrite(img_name + "_logCorr" + ".jpg", new_img_logCorr)
# Grayscale
new_grayscale_img_logCorr = correction.logarithm_correction(
    True, new_grayscale_img_compressed
)
cv.imwrite(img_name + "_gray" + "_logCorr" + ".jpg", new_grayscale_img_logCorr)
# Histograms
plt = plot_creator.get_plot(
    "Histograms of brightness after logarithm correction",
    new_img_logCorr,
    new_grayscale_img_logCorr,
)
print("End of Task 3")
plt.show()
