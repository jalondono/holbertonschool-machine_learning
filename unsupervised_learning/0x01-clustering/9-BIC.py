#!/usr/bin/env python3
""" Full BIC"""

import numpy as np
expectation_maximization = __import__('7-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the Bayesian
    Information Criterion:
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param kmin: is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    :param kmax: is a positive integer containing the maximum number of
     clusters to check for (inclusive)
    :param iterations: is a positive integer containing the maximum
    number of iterations for the EM algorithm
    :param tol: is a non-negative float containing the tolerance for
    the EM algorithm
    :param verbose: is a boolean that determines if the EM algorithm
     should print information to the standard output
    :return: best_k, best_result, l, b, or None, None, None, None on failure
    * best_k is the best value for k based on its BIC
    * best_result is tuple containing pi, m, S
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) != int or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or X.shape[0] < kmax:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    n, d = X.shape

    b = []
    results = []
    ks = []
    l_ = []

    for k in range(kmin, kmax + 1):
        ks.append(k)

        pi, m, S, g, l_k = expectation_maximization(X,
                                                    k,
                                                    iterations=iterations,
                                                    tol=tol,
                                                    verbose=verbose)
        results.append((pi, m, S))

        l_.append(l_k)
        p = k - 1 + k * d + k * d * (d + 1) / 2

        bic = p * np.log(n) - 2 * l_k
        b.append(bic)

    l_ = np.array(l_)
    b = np.array(b)

    index = np.argmin(b)
    best_k = ks[index]
    best_result = results[index]

    return best_k, best_result, l_, b
