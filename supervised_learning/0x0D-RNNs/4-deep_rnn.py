#!/usr/bin/env python3
"""Deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a deep RNN:
    :param rnn_cells: is a list of RNNCell instances of length l
    that will be used for the forward propagation
    :param X: is the data to be used, given as a numpy.ndarray
    of shape (t, m, i)
    :param h_0: is the initial hidden state, given as a
     numpy.ndarray of shape (l, m, h)
    :return: H, Y
    """
    # shapes of arrays
    steps = X.shape[0]
    l, m, h = h_0.shape

    # initialization of arrays
    H = np.empty((steps + 1, l, m, h))
    H[0] = h_0
    Y = []
    y_aux = []

    for i in range(steps):
        Xo = np.copy(X[i])
        for idx, cell in enumerate(rnn_cells):
            H[i+1][idx], y_aux = cell.forward(H[i][idx], Xo)
            Xo = H[i+1][idx]
        Y.append(y_aux)

    return H, np.array(Y)
