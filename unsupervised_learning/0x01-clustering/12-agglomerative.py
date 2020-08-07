#!/usr/bin/env python3
"""Agglomerative"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset:
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
    :param dist: is the maximum cophenetic distance for all clusters
    :return: clss, a numpy.ndarray of shape (n,) containing the cluster
     indices for each data point
    """
    h = scipy.cluster.hierarchy
    Z = h.linkage(X, 'ward')
    ind = h.fcluster(Z, t=dist, criterion="distance")
    fig = plt.figure()
    dn = h.dendrogram(Z, color_threshold=dist)
    plt.show()
    return ind