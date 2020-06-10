#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape,
                 stride=(1, 1), mode='max'):
    """
    performs forward propagation over a convolutional
     layer of a neural network:

    :param A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
     containing the output of the previous layer
    :param kernel_shape: is a tuple of (kh, kw) containing the size of
     the kernel for the pooling
    :param stride:is a tuple of (sh, sw)
     containing the strides for the convolution
    :param mode: is a string containing either max or avg, indicating whether
     to perform maximum or average pooling, respectively
    :return: the output of the convolutional layer
    """
    # image sizes
    m = A_prev.shape[0]
    h = A_prev.shape[1]
    w = A_prev.shape[2]
    c = A_prev.shape[3]

    # Kernel Size
    kh = kernel_shape[0]
    kw = kernel_shape[1]

    # Strides
    sh = stride[0]
    sw = stride[1]

    # output size
    out_h = int(((h - kh) / sh) + 1)
    out_w = int(((w - kw) / sw) + 1)

    out_img = np.zeros((m, out_h, out_w, c))
    for x in range(out_w):
        for y in range(out_h):
            if mode == 'max':
                img = A_prev[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
                pixel = np.max(img, axis=(1, 2))
                out_img[:, y, x, :] = pixel
            if mode == 'avg':
                img = A_prev[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
                pixel = np.mean(img, axis=(1, 2))
                out_img[:, y, x, :] = pixel
    return out_img