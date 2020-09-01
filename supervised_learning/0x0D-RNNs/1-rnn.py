#!/usr/bin/env python3
"""RNN Fordward pass"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation for a simple RNN:
    :param rnn_cell: is an instance of RNNCell that
     will be used for the forward propagation
    :param X: is the data to be used, given as a numpy.ndarray
     of shape (t, m, i)
    :param h_0: is the initial hidden state, given as a
     numpy.ndarray of shape (m, h)
    :return: H, Y
    """
    t = X.shape[0]
    m, h = h_0.shape

    H = np.empty((t+1, m, h))
    Y = []
    H[0] = h_0
    for i in range(t):
        h_prev = H[i]
        x_t = X[i]
        H[i+1], y_aux = rnn_cell.forward(h_prev, x_t)
        Y.append(y_aux)
    return H, np.array(Y)
