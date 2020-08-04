#!/usr/bin/env python3
"""Perform K-means"""

import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param C: is a numpy.ndarray of shape (k, d) containing the centroid
     means for each cluster
    :return: var, or None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(X.shape) != 2:
            return None
        data = X[:, :, np.newaxis]
        centroids = C.T[np.newaxis, :, :]
        diff = (centroids - data)
        dist = np.linalg.norm(diff, axis=1)
        clss = np.min(dist, axis=1)
        variance = np.sum(clss ** 2)
        return np.sum(variance)
    except Exception:
        return None
