import os.path
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import torchaudio.transforms as T
import torchvision.transforms as trans
import cv2


def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return rst


def image_darkaug(img,img_label,dark_aug,brightness=1,conv2d=0):
    if dark_aug==1:
        creterion = random.random()
        # print(creterion)
        if creterion>0.6:
            brightness = 0
            img = imgBrightness(img,brightness,3)
            img_label = np.array([0])

    elif dark_aug==2:
            img = imgBrightness(img,brightness,3)
            img_label = np.array([0])

    elif dark_aug>2:
        brightness = 0.04/dark_aug
        img = imgBrightness(img,brightness,3)
        img_label = np.array([0])

    img = preprocess_input(img)
    return img,img_label,brightness




def image_darkaug_test(img,dark_aug,brightness=1,conv2d=0):
    if dark_aug==1:
        creterion = random.random()
        # print(creterion)
        if creterion>0.6:
            brightness = 0
            img = imgBrightness(img,brightness,3)
            img_label = np.array([0])

    elif dark_aug==2:
            img = imgBrightness(img,brightness,3)
            img_label = np.array([0])

    elif dark_aug>2:
        brightness = 0.04/dark_aug
        img = imgBrightness(img,brightness,3)
        img_label = np.array([0])

    img = preprocess_input(img)
    return img,brightness
def preprocess_input(image):
    image = image / 127.5-1
    return image

