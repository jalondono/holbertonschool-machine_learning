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
    # (m, h_new, w_new, c_new) = dA.shape
    #
    # # image sizes
    # m = A_prev.shape[0]
    #
    # # Kernel Size
    # kh = kernel_shape[0]
    # kw = kernel_shape[1]
    #
    # # Strides
    # sh = stride[0]
    # sw = stride[1]
    #
    # dA = np.zeros_like(A_prev, dtype=dA.dtype)
    #
    # for i in range(m):
    #     for y in range(h_new):
    #         for x in range(w_new):
    #             for ch in range(c_new):
    #                 pool = A_prev[i, y * sh: y * sh + kh, x * sw: x * sw + kw, ch]
    #                 aux_dA = dA[i, y, x, ch]
    #
    #                 if mode == 'max':
    #                     mask = create_mask_from_window(pool)
    #                     dA[i, y * sh: y * sh + kh, x * sw: x * sw + kw, ch] += mask * aux_dA
    #
    #                 if mode == 'average':
    #                     mask = np.ones(kernel_shape)
    #                     avg = aux_dA / kh / kw
    #                     dA[i, y * sh: y * sh + kh, x * sw: x * sw + kw: ch] += \
    #                         mask * avg
    # return dA
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)
    for m_i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c_i in range(c):
                    pool = A_prev[m_i, h * sh:(kh + h * sh), w * sw:(kw + w * sw), c_i]
                    dA_val = dA[m_i, h, w, c_i]
                    if mode == 'max':
                        zero_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(zero_mask, pool == _max, 1)
                        dA_prev[m_i, h * sh:(kh + h * sh),
                        w * sw:(kw + w * sw), c_i] += zero_mask * dA_val
                    if mode == 'avg':
                        avg = dA_val / kh / kw
                        one_mask = np.ones(kernel_shape)
                        dA_prev[m_i, h * sh:(kh + h * sh),
                        w * sw:(kw + w * sw), c_i] += one_mask * avg
    return dA_prev
