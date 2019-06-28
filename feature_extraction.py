import cv2
import numpy as np


def ReadImage(path):
    img = cv2.imread(path,  cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img,-1)/255
    return img


def ExtractClass(img_path):
    y_train=img_path.split('digit')[1].split('.')[0]
    return y_train

