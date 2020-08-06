#!/usr/bin/env python3
""" Initialize PDF """

import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution:
    :param X: is a numpy.ndarray of shape (n, d) containing the data
    points whose PDF should be evaluated
    :param m: is a numpy.ndarray of shape (d,) containing
    the mean of the distribution
    :param S: is a numpy.ndarray of shape (d, d) containing
    the covariance of the distribution
    :return: P, or None on failure
    * P is a numpy.ndarray of shape (n,) containing
    the PDF values for each data point
    * All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    n, d = X.shape

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    exp_1 = (1 / (((2 * np.pi) ** (d / 2)))) * (det ** (-1 / 2))
    exp_2 = ((-1/2)*(X-m))
    exp_3 = np.matmul(exp_2, inv)
    exp_4 = np.sum(exp_3.T * (X-m).T, axis=0)
    result = np.exp(exp_4) * exp_1
    result = np.where(result < 1e-300, 1e-300, result)
    return result
