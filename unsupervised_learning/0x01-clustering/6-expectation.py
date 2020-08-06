#!/usr/bin/env python3
""" Initialize PDF """

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates the expectation step in the EM algorithm for a GMM:
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param pi: is a numpy.ndarray of shape (k,) containing the priors for
    each cluster
    :param m: is a numpy.ndarray of shape (k, d) containing the centroid
     means for each cluster
    :param S: is a numpy.ndarray of shape (k, d, d) containing the covariance
     matrices for each cluster
    :return: g, l, or None, None on failure
    * g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities
    for each data point in each cluster
    * l is the total log likelihood
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    k = pi.shape[0]
    n, d = X.shape

    if m.shape[1] != d or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if S.shape[0] != k:
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    posterior = np.zeros((k, n))

    for idx in range(k):
        posterior[idx] = pi[idx] * pdf(X, m[idx], S[idx])
    marginal = np.sum(posterior, axis=0)
    posterior = posterior/marginal
    log_likelihood = np.sum(np.log(marginal))
    return posterior, log_likelihood
