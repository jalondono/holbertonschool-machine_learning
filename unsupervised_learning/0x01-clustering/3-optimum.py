#!/usr/bin/env python3
"""Perform K-means"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum number of clusters by variance:
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param kmin: is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    :param kmax: is a positive integer containing the maximum number of
    clusters to check for (inclusive)
    :param iterations: is a positive integer containing the maximum number of
    iterations for K-means
    :return: results, d_vars, or None, None on failure
    * results is a list containing the outputs of K-means for each cluster size
    * d_vars is a list containing the difference in variance from the smallest
    cluster size for each cluster size
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if type(kmin) != int or kmin <= 0 or X.shape[0] <= kmin:
            return None, None
        if type(kmax) != int or kmax <= 0 or X.shape[0] < kmax:
            return None, None
        if kmax <= kmin:
            return None, None
        if type(iterations) != int or iterations <= 0:
            return None, None

        results = []
        var_k = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))
            var = variance(X, C)
            var_k.append(var)
        d_vars = [var_k[0] - var for var in var_k]
        return results, d_vars
    except Exception:
        return None, None
