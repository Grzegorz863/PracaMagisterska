import os
import cv2
from os.path import join

import numpy as np


def read_my_images(path):
    my_imgs = []
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = cv2.resize(cv2.imread(join(path, f)), (100, 100))
        my_imgs.append(img)

    return np.asarray(my_imgs, dtype=np.float32)
