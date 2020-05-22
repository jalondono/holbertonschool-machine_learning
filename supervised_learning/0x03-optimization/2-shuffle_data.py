#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way:
    :param X: is the first numpy.ndarray of shape (m, nx) to shuffle
    :param Y: is the second numpy.ndarray of shape (m, ny) to shuffle
    :return:
    """
    x = np.random.permutation(X)
    y = np.random.permutation(Y)
    return x, y
