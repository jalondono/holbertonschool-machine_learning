#!/usr/bin/env python3
"""GRU Cell """

import numpy as np


class GRUCell:
    """GRUCell class"""

    def __init__(self, i, h, o):
        """
        Constructor method
        :param i: is the dimensionality of the data
        :param h: is the dimensionality of the hidden state
        :param o: is the dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, Z):
        """
        Sigmoid activation function
        :param Z: is the array of W.X + b values
        :return: Y predicted
        """
        sigma = (1 / (1 + np.exp(-Z)))
        return sigma

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev:  is a numpy.ndarray of shape (m, i)
         that contains the data input for the cell
        :param x_t: is a numpy.ndarray of shape (m, h)
         containing the previous hidden state
        :return: h_next, y
        """
        conc_input = np.concatenate((h_prev, x_t), axis=1)

        # update gqate
        zt = self.sigmoid(np.matmul(conc_input, self.Wz) + self.bz)

        # reset gate
        rt = self.sigmoid(np.matmul(conc_input, self.Wr) + self.br)

        h_x = np.concatenate((rt * h_prev, x_t), axis=1)
        # new candidates
        ht_hat = np.tanh(np.matmul(h_x, self.Wh) + self.bh)

        # Final memory
        ht = (1 - zt) * h_prev + zt * ht_hat

        Z_y = np.matmul(ht, self.Wy) + self.by

        # softmax activation
        y = self.softmax(Z_y)

        return ht, y
