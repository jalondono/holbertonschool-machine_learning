#!/usr/bin/env python3
"""Convolution with Padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding:
    :param images: is a numpy.ndarray with shape (m, h, w)
     containing multiple grayscale images
    :param kernel: is a numpy.ndarray with shape (kh, kw)
     containing the kernel for the convolution
    :param padding: is a tuple of (ph, pw)
    :return:
    """
    # image sizes
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    # Padding size
    pph = padding[0]
    ppw = padding[1]

    # Kernel Size
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # New image size
    new_h = h + 2 * pph - kh + 1
    new_w = w + 2 * ppw - kw + 1

    # Padding to example images
    ph = int(pph / 1)
    pw = int(ppw / 1)

    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    out_img = np.zeros((m, new_h, new_w))
    for x in range(new_w):
        for y in range(new_h):
            img = images_padded[:, y: y + kh, x: x + kw]
            pixel = np.sum(img * kernel, axis=(1, 2))
            out_img[:, y, x] = pixel
    return out_img
