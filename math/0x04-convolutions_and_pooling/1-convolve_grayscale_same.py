#!/usr/bin/env python3
"""Valid Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a valid convolution on grayscale images:
    :param images: is a numpy.ndarray with shape (m, h, w) containing
    multiple
    :param kernel: is a numpy.ndarray with shape (kh, kw) containing
     the kernel for the convolution
    :return: a numpy.ndarray containing the convolved image
    """
    w = images.shape[2]
    h = images.shape[1]
    m = images.shape[0]

    kw = kernel.shape[0]
    kh = kernel.shape[1]

    ph = int((kh - 1) / 2)
    pw = int((kw - 1) / 2)

    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    out_img = np.zeros((m, w, h))

    for x in range(w):
        for y in range(h):
            img = images_padded[:, y: y + kh, x: x + kw]
            pixel = np.sum(img * kernel, axis=(1, 2))
            out_img[:, y, x] = pixel
    return out_img
