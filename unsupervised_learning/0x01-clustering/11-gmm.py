#!/usr/bin/env python3
""" Hello, sklearn!"""

import sklearn.mixture


def gmm(X, k):
    """
     calculates a GMM from a dataset:
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
    :param k: is the number of clusters
    :return: pi, m, S, clss, bic
    """
    if len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    G = GMM.fit(X)
    pi = G.weights_
    m = G.means_
    S = G.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)
    return pi, m, S, clss, bic
