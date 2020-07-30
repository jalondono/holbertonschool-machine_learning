#!/usr/bin/env python3
""" PCA function"""
import numpy as np


def pca(X, var=0.95):
    """
    function def pca(X, var=0.95): that performs PCA on a dataset:
    :param X: is a numpy.ndarray of shape (n, d) where:
    :param var: is the fraction of the variance that the PCA
    transformation should maintain
    :return:the weights matrix, W, that maintains var fraction
    of X‘s original variance
    """
    u, s, vh = np.linalg.svd(X)
    accum = np.cumsum(s)
    threshold = accum[-1] * var
    ndim = len(list(filter(lambda x: x < threshold, accum)))
    W = vh.T[:, :ndim + 1]
    return W
