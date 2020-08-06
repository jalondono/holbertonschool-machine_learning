#!/usr/bin/env python3
""" Initialize GMM """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
     initializes variables for a Gaussian Mixture Model:
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param k: is a positive integer containing the number of clusters
    :return: pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None, None

    n, d = X.shape
    pi = np.tile(1/k, (k,))
    m, _ = kmeans(X, k)
    identity = np.identity(d)
    S = np.tile(identity, (k, 1, 1))

    return pi, m, S
