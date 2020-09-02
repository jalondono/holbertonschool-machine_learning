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

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        :param h_prev:  is a numpy.ndarray of shape (m, i)
         that contains the data input for the cell
        :param x_t: is a numpy.ndarray of shape (m, h)
         containing the previous hidden state
        :return: h_next, y
        """
