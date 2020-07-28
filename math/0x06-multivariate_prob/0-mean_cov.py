#!/usr/bin/env python3
"""Mean and Covariance """
import numpy as np


def mean_cov(X):
    """
    calculates the mean and covariance of a data set:
    :param X:
    :return:
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError(' must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m
    mean = mean[np.newaxis, :]

    dev = X - mean
    cov = np.matmul(dev.T, dev) / (m - 1)
    return mean, cov
