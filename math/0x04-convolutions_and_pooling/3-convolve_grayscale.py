#!/usr/bin/env python3
"""Strided Convolution """

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images:
    :param images: is a numpy.ndarray with shape (m, h, w)
     containing multiple grayscale images
    :param kernel: is a numpy.ndarray with shape (kh, kw)
     containing the kernel for the convolution
    :param padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
    :param stride: is a tuple of (sh, sw)
    :return: a numpy.ndarray containing the convolved images
    """
    # image sizes
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    # Kernel Size
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # Strides
    sh = stride[0]
    sw = stride[1]

    # padding
    ph = 0
    pw = 0

    if padding == 'same':
        # padding of zeros
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    out_h = int(((h - 2 * ph - kh) / sh) + 1)
    out_w = int(((w - 2 * pw - kw) / sw) + 1)

    img_padded = np.pad(images,
                        pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)

    out_img = np.zeros((m, out_h, out_w))
    for x in range(out_w):
        for y in range(out_h):
            img = img_padded[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
            pixel = np.sum(img * kernel, axis=(1, 2))
            out_img[:, y, x] = pixel
    return out_img
