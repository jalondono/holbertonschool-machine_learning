#!/usr/bin/env python3
"""Bidirectional Output"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:
    :param X: is the data to be used, given as a numpy.ndarray
    of shape (t, m, i)
    :param h_0: is the initial hidden state in the forward direction,
     given as a numpy.ndarray of shape (m, h)
    :param h_t: h_t is the initial hidden state in the backward direction,
    given as a numpy.ndarray of shape (m, h)
    :return: H, Y
    """
    t, m, i = X.shape

    H_f = []
    H_b = []

    # initialization
    h_f = h_0
    h_b = h_t

    H_f.append(h_0)
    H_b.append(h_t)

    # traverse inputs
    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        h_b = bi_cell.backward(h_b, X[t - 1 - step])

        H_f.append(h_f)
        H_b.append(h_b)

    H_f = np.array(H_f)
    H_b = [x for x in reversed(H_b)]
    H_b = np.array(H_b)
    H = np.concatenate((H_f[1:], H_b[:-1]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
