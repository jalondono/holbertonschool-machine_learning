#!/usr/bin/env python3
"""Valid Convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images:
    :param images: is a numpy.ndarray with shape (m, h, w) containing
    multiple
    :param kernel: is a numpy.ndarray with shape (kh, kw) containing
     the kernel for the convolution
    :return: a numpy.ndarray containing the convolved image
    """
    w = int(images.shape[2] - kernel.shape[1] + 1)
    h = int(images.shape[1] - kernel.shape[0] + 1)
    m = int(images.shape[0])

    kw = kernel.shape[1]
    kh = kernel.shape[0]
    new_img = np.zeros((m, h, w))

    for row in range(w):
        for colum in range(h):
            img = images[0:, colum: kh + colum, row: kw + row]
            pixel = np.sum(img * kernel, axis=(1, 2))
            new_img[0:, colum, row] = pixel
    return new_img
