#!/usr/bin/env python3
"""Normalization Constants """
import numpy as np


def normalize(X, m, s):
    """

    :param X: is the numpy.ndarray of shape (d, nx) to normalize
    d is the number of data points
    nx is the number of features
    :param m: is a numpy.ndarray of shape (nx,) that contains the mean
     of all features of X
    :param s: is a numpy.ndarray of shape (nx,) that contains the
     standard deviation of all features of X
    :return: Returns: The normalized X matrix
    """

    return (X - m) / s
