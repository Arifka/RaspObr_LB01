import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt

def compression_image(flag: bool, img, min=50, max=200):
    alpha = 1.0
    beta = 0.0
    new_img_compressed = np.zeros(img.shape, img.dtype)
    match flag:
        case 0:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    for channel in range(img.shape[2]):
                        new_img_compressed[y, x, channel] = np.clip(alpha*img[y, x, channel]+beta, min, max)
        case 1:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    new_img_compressed[y, x] = np.clip(alpha*img[y, x]+beta, min, max)
    return new_img_compressed

def gamma_correction(flag: bool, img):
    coef_c = 1.55
    coef_gamma = 2
    new_img_gammaCorr = np.zeros(img.shape, img.dtype)
    match flag:
        case 0:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    for c in range(img.shape[2]):
                        new_img_gammaCorr[y, x, c] = 255 * coef_c * (img[y, x, c]/255) ** coef_gamma
        case 1:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    new_img_gammaCorr[y, x] = 255 * coef_c * (img[y, x] / 255) ** coef_gamma
    return new_img_gammaCorr

def logarithm_correction(flag: bool, img):
    coef_c = 1.55
    new_img_logCorr = np.zeros(img.shape, img.dtype)
    match flag:
        case 0:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    for c in range(img.shape[2]):
                        new_img_logCorr[y, x, c] = 255 * coef_c * math.log((img[y, x, c] / 255) + 1)
        case 1:
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    new_img_logCorr[y, x] = 255 * coef_c * math.log((img[y, x] / 255) + 1)
    return new_img_logCorr

def linier_correction(flag: bool, img):
    new_img_linCorr = np.zeros(img.shape, img.dtype)
    match flag:
        case 0:
            b_min = img[0, 0, 0]
            r_min = img[0, 0, 1]
            g_min = img[0, 0, 2]
            list_min = [b_min, r_min, g_min]

            b_max = img[0, 0, 0]
            r_max = img[0, 0, 1]
            g_max = img[0, 0, 2]
            list_max = [b_max, r_max, g_max]
            #Find min, max of BRG pixel brightness
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    for c in range(img.shape[2]):
                        list_min[c] = min(list_min[c], img[y, x, c])
                        list_max[c] = max(list_max[c], img[y, x, c])
            # Change brightness with formule            
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    for channel in range(img.shape[2]):
                        new_img_linCorr[y, x, channel] = (img[y, x, channel] - list_min[channel]) * 255 / (list_max[channel] - list_min[channel])
        case 1:
            brightness_min = img[0, 0]
            brightness_max = img[0, 0]
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    brightness_min = min(brightness_min, img[y, x])
                    brightness_max = max(brightness_max, img[y, x])
            # Change brightness with formule
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    new_img_linCorr[y, x] = (img[y, x] - brightness_min)*255/(brightness_max - brightness_min)
            # Save result
    return new_img_linCorr