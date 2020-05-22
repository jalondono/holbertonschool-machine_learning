#!/usr/bin/env python3
"""Normalization Constants """
import numpy as np


def normalization_constants(X):
    """
    alculates the normalization (standardization) constants of a matrix:
    :param X: is the numpy.ndarray of shape (m, nx) to normalize
    :return: the mean and standard deviation of each feature, respectively
    """
    size = X.shape[0]
    mean = X.sum(axis=0) / size
    sigma = abs((X - mean) ** 2)
    sigma_mean = sigma.sum(axis=0) / size
    sigma_end = np.sqrt(sigma_mean)
    return mean, sigma_end
    #
