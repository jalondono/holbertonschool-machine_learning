#!/usr/bin/env python3
""" Hello, sklearn!"""

import sklearn.cluster
import numpy as np


def kmeans(X, k):
    """
    performs K-means on a dataset:
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
    :param k: is the number of clusters
    :return: C, clss
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None

    Kmean = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = Kmean.cluster_centers_
    clss = Kmean.labels_
    return C, clss
