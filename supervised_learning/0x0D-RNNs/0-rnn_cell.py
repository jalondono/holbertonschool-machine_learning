#!/usr/bin/env python3
"""RNN Cell"""

import numpy as np


class RNNCell:
    """RNN class"""
    def __init__(self, i, h, o):
        """
        constructor method
        :param i: is the dimensionality of the data
        :param h: is the dimensionality of the hidden state
        :param o: is the dimensionality of the outputs
        """
        self.Wh = np.random.randn(i+h, h)
        self.by = np.zeros((1, o))
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))

    def softmax(self, x):
        """ softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
         performs forward propagation for one time step
        :param h_prev: is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        :param x_t: is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell
        :return: h_next, y
        * h_next is the next hidden state
        * y is the output of the cell
        """
        # using h(t) or next_t = tanh(Wh[x^(t);h^(t-1)] + bh
        a = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(a, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)
        return h_next, y
