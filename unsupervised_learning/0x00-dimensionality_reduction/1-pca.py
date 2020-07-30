#!/usr/bin/env python3
import numpy as np


def pca(X, ndim):
    """
    function def pca(X, ndim): that performs PCA on a dataset:
    :param X: is a numpy.ndarray of shape (n, d) where:
    :param ndim: is the new dimensionality of the transformed X
    :return: T, a numpy.ndarray of shape (n, ndim) containing
     the transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh.T[:, :ndim]
    T = np.matmul(X_m, W)
    return T
