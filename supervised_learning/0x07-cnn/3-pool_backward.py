#!/usr/bin/env python3
""" Pooling Back Prop"""
import numpy as np


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position
     corresponding to the max entry of x.
    """

    mask = np.zeros(x.shape)
    np.place(mask, x == np.amax(x), 1)
    return mask


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:
    :param dA:
    :param A_prev:
    :param kernel_shape:
    :param stride:
    :param mode:
    :return:
    """
    (m, h_new, w_new, c_new) = dA.shape

    # image sizes
    m = A_prev.shape[0]

    # Kernel Size
    kh = kernel_shape[0]
    kw = kernel_shape[1]

    # Strides
    sh = stride[0]
    sw = stride[1]

    dA = np.zeros_like(A_prev, dtype=dA.dtype)

    for i in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for ch in range(c_new):
                    pool = A_prev[i, y * sh: y * sh + kh, x * sw: x * sw + kw, ch]
                    aux_dA = dA[i, y, x, ch]

                    if mode == 'max':
                        mask = create_mask_from_window(pool)
                        dA[i, y * sh: y * sh + kh, x * sw: x * sw + kw, ch] += mask * aux_dA

                    if mode == 'average':
                        mask = np.ones(kernel_shape)
                        avg = aux_dA / kh / kw
                        dA[i, y * sh: y * sh + kh, x * sw: x * sw + kw: ch] += \
                            mask * avg
    return dA
