import cv2 as cv
from matplotlib import pyplot as plt


def get_plot(name: str, img, grayscale_img):
    plot = plt
    plot.figure(name)
    plot.subplot(211)
    plot = create_histograms(False, img)
    plot.subplot(212)
    plot = create_histograms(True, grayscale_img)
    plot.tight_layout()
    return plot


def create_histograms(flag: bool, img):  # Тип изображения: 0 - BRG, 1 - GrayScale
    match flag:
        case 0:
            color = ("b", "g", "r")
            for i, col in enumerate(color):
                histr = cv.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
        case 1:
            histr = cv.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(histr)
            plt.xlim([0, 256])
    return plt
