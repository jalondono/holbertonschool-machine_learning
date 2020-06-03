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
    w = images.shape[2] - (kernel.shape[1] - 1)
    h = images.shape[1] - (kernel.shape[1] - 1)
    m = images.shape[0]

    kw = kernel.shape[0]
    kh = kernel.shape[1]
    new_img = np.zeros((m, w, h))

    for row in range(w):
        for colum in range(h):
            img = images[0:, row: kw + row, colum: kh + colum]
            pixel = np.sum(img * kernel, axis=(1, 2))
            new_img[0:, row, colum] = pixel
    return new_img
