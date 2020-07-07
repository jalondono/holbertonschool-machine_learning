#!/usr/bin/env python3
""" Load Images """
import glob
import cv2
import numpy as np


def load_images(images_path, as_array=True):
    """
    loads images from a directory or file:
    :param images_path: is the path to a directory from which to load images
    :param as_array:is a boolean indicating whether the images should be
     loaded as one numpy.ndarray
    :return: Returns: images, filenames
    """
    imgs = []
    names = []
    paths = glob.glob(images_path + '/*')
    paths = sorted([i for i in paths])
    for path in paths:
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        names.append(path[len(images_path) + 1:])
    if not as_array:
        return imgs, names
    else:
        imgs = np.array(imgs)
        return imgs, names
