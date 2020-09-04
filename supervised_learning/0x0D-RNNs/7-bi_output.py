#!/usr/bin/env python3
"""Bidirectional Output"""

import numpy as np


class BidirectionalCell:
    """ BidirectionalCell class"""
    def __init__(self, i, h, o):
        """
        class constructor
        :param i: is the dimensionality of the data
        :param h: is the dimensionality of the hidden states
        :param o: is the dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """ softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        represents a bidirectional cell of an RNN:
        :param h_prev: is a numpy.ndarray of shape (m, h)
         containing the previous hidden state
        :param x_t: is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell
        :return: h_next, the next hidden state
        """
        a = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(a, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction for one time step
        :param h_next: numpy.ndarray of shape (m, h)
            containing the next hidden state
        :param x_t: numpy.ndarray of shape (m, i)
            that contains the data input for the cell
            m is the batch size for the data
        :return: h_pev, the previous hidden state
        """

        h_x = np.concatenate((h_next, x_t), axis=1)
        Z_prev = np.matmul(h_x, self.Whb) + self.bhb
        h_prev = np.tanh(Z_prev)

        return h_prev

    def output(self, H):
        """
        calculates all outputs for the RNN:
        :param H: is a numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions, excluding
        their initialized states
        :return: Y, the outputs
        """
        t = H.shape[0]
        y = []

        for idx in range(t):
            y.append(self.softmax(np.matmul(H[idx], self.Wy) + self.by))
        return np.array(y)
