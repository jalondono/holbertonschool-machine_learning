#!/usr/bin/env python3
"""  Convolution with Channels """

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images:
    :param images: is a numpy.ndarray with shape (m, h, w)
     containing multiple grayscale images
    :param kernels:  is a numpy.ndarray with shape (kh, kw, c, nc)
     containing the kernels for the convolution
     containing the kernel for the convolution
    :param padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
    :param stride: is a tuple of (sh, sw)
    :return: a numpy.ndarray containing the convolved images
    """
    # image sizes
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    n_k = kernels.shape[3]

    # Kernel Size
    kh = kernels.shape[0]
    kw = kernels.shape[1]

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

    out_h = int(((h - kh + (2 * ph)) / sh) + 1)
    out_w = int(((w - kw + (2 * pw)) / sw) + 1)

    img_padded = np.pad(images,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    out_img = np.zeros((m, out_h, out_w, n_k))
    for x in range(out_w):
        for y in range(out_h):
            for idx in range(n_k):
                img = img_padded[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
                pixel = np.sum(img * kernels[:, :, :, idx], axis=(1, 2, 3))
                out_img[:, y, x, idx] = pixel
    return out_img
