#!/usr/bin/env python3
""" Full EM"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    performs the expectation maximization for a GMM:
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param k: is a positive integer containing the number of clusters
    :param iterations: is a positive integer containing the maximum number
     of iterations for the algorithm
    :param tol: is a non-negative float containing tolerance of the log
     likelihood, used to determine early stopping i.e. if the difference
      is less than or equal
     to tol you should stop the algori
    :param verbose: is a boolean that determines if you should print
     information about the algorithm
    :return:pi, m, S, g, l, or None, None, None, None, None on failure
    """
    # conditions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    tol_off = False
    pi, m, S = initialize(X, k)
    g, lh = expectation(X, pi, m, S)
    prev_l = 0

    for i in range(iterations + 1):
        if i % 10 == 0 and verbose:
            print('Log Likelihood after {} '
                  'iterations: {}'.format(i, lh.round(5)))
        if i != iterations:
            pi, m, S = maximization(X, g)
            g, lh = expectation(X, pi, m, S)
            if abs(lh - prev_l) <= tol:
                break
            prev_l = lh
    if verbose:
        print('Log Likelihood after {} '
              'iterations: {}'.format(i+1, lh.round(5)))
    return pi, m, S, g, lh
