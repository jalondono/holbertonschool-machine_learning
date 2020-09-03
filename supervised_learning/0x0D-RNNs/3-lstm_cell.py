#!/usr/bin/env python3
"""LSTM Cell """

import numpy as np


class LSTMCell:
    """LSTMCell Class"""
    def __init__(self, i, h, o):
        """
        Constructor method
        :param i: is the dimensionality of the data
        :param h: is the dimensionality of the hidden state
        :param o: is the dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, o))
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

    def forward(self, h_prev, c_prev, x_t):
        """

        :param h_prev: is a numpy.ndarray of shape (m, i) that contains
         the data input for the cell
        :param c_prev: is a numpy.ndarray of shape (m, h) containing
         the previous hidden state
        :param x_t: is a numpy.ndarray of shape (m, h) containing
         the previous cell state
        :return: h_next, c_next, y
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        ft = self.sigmoid(np.matmul(concat_input, self.Wf) + self.bf)

        # update Gate
        it = self.sigmoid(np.matmul(concat_input, self.Wu) + self.bu)

        # New candidates
        ct_hat = np.tanh(np.matmul(concat_input, self.Wc) + self.bc)

        # output gate
        ot = self.sigmoid(np.matmul(concat_input, self.Wo) + self.bo)

        # next cell memory
        ct = (ft * c_prev) + (it * ct_hat)

        # next hidden state
        ht = ot * np.tanh(ct)

        y = self.softmax(np.matmul(ht, self.Wy) + self.by)
        return ht, ct, y
