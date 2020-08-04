#!/usr/bin/env python3
"""Perform K-means"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset:
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
    :param k:  is a positive integer containing the number of clusters
    :param iterations:is a positive integer containing the maximum number
     of iterations that should be performed
    :return: C, clss, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    n, d = X.shape

    min_value = np.amin(X, axis=0)
    max_value = np.amax(X, axis=0)

    cents = np.random.uniform(min_value, max_value, (k, d))
    cents_pev = np.copy(cents)
    centroids_axis = cents.T[np.newaxis, :, :]

    X_axis = X[:, :, np.newaxis]

    diff = X_axis - centroids_axis
    eucli_dist = np.linalg.norm(diff, axis=1)
    clss = np.argmin(eucli_dist, axis=1)

    for i in range(iterations):
        for j in range(k):
            index = np.where(j == clss)
            if len(index[0]) == 0:
                cents[j] = np.random.uniform(min_value, max_value, (1, d))
            else:
                cents[j] = np.mean(X[index], axis=0)

        centroids_axis = cents.T[np.newaxis, :, :]
        X_axis = X[:, :, np.newaxis]
        diff = X_axis - centroids_axis
        eucli_dist = np.linalg.norm(diff, axis=1)
        clss = np.argmin(eucli_dist, axis=1)

        if (cents == cents_pev).all():
            return cents, clss
        cents_pev = np.copy(cents)
    return cents, clss
