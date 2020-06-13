#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional
     layer of a neural network:
    :param A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
     containing the output of the previous layer
    :param W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
     containing the kernels for the convolution
    :param b: is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    :param activation: is an activation function applied to the convolution
    :param padding:is a string that is either same or valid,
     indicating the type of padding used
    :param stride:is a tuple of (sh, sw)
     containing the strides for the convolution
    :return: the output of the convolutional layer
    """
    # image sizes
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    # Kernel Size
    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]

    # Strides
    sh = stride[0]
    sw = stride[1]

    # padding
    ph = 0
    pw = 0

    if padding == 'same':
        if kh % 2 == 0:
            ph = int((h_prev * sh + kh - h_prev) / 2)
            n_h = int(((h_prev + 2 * ph - kh) / sh))
        else:
            ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
            n_h = int(((h_prev + 2 * ph - kh) / sh) + 1)

        if kw % 2 == 0:
            pw = int((w_prev * sw + kw - w_prev) / 2)
            n_w = int(((w_prev + 2 * pw - kw) / sw))
        else:
            pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
            n_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    out_h = int(((h_prev - kh + (2 * ph)) / sh) + 1)
    out_w = int(((w_prev - kw + (2 * pw)) / sw) + 1)

    img_padded = np.pad(A_prev,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    out_img = np.zeros((m, n_h, n_w, c_new))
    for x in range(out_w):
        for y in range(out_h):
            for idx in range(c_new):
                img = img_padded[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
                pixel = np.sum(img * W[:, :, :, idx], axis=(1, 2, 3))
                out_img[:, y, x, idx] = pixel
    z = out_img + b
    return activation(z)
