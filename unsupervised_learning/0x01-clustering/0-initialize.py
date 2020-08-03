#!/usr/bin/env python3
"""Initialize K-means """

import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:
    :param X:  is a numpy.ndarray of shape (n, d) containing the
    dataset that will be used for K-means clustering
    :param k: is a positive integer containing the number of clusters
    :return:  a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    d = X.shape[1]
    centroids = np.random.uniform(min, max, (k, d))
    return centroids
