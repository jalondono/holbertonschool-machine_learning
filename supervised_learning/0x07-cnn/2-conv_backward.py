#!/usr/bin/env python3
"""convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
     performs back propagation over a convolutional
      layer of a neural network:
    :param dZ: is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
     the partial derivatives with respect to the unactivated
      output of the convolutional layer
    :param A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
     containing the output of the previous layer
    :param W:is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
     containing the kernels for the convolution
    :param b:is a numpy.ndarray of shape (1, 1, 1, c_new)
     containing the biases applied to the convolution
    :param padding: is a string that is either same or valid,
     indicating the type of padding used
    :param stride: is a tuple of (sh, sw)
     containing the strides for the convolution
    :return:the partial derivatives with respect to
     the previous layer (dA_prev),
     the kernels (dW), and the biases (db), respectively
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    # Retrieving dimensions from dZ
    (m, h_new, w_new, c_new) = dZ.shape

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
        # padding of zeros
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    A_prev_Padded = np.pad(A_prev,
                           pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    dA = np.zeros(A_prev_Padded.shape)
    dw = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Go through all examples
    for i in range(m):
        for x in range(w_new):
            for y in range(h_new):
                for c in range(c_new):
                    dA[:, y * sh: y * sh + kh, x * sw: x * sw + kw]\
                        += W[:, :, :, c] * dZ[i, y, x, c]
                    aux_str = 'A_prev[i, y * sh: y *' \
                              ' sh + kh, x * sw: x * sw + kw]'
                    dw[:, :, :, c]\
                        += eval(aux_str) * dZ[i, y, x, c]
    return dA, dw, db
