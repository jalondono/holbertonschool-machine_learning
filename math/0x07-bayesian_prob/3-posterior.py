#!/usr/bin/env python3
"""Posterior Probability"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data:
    :param x:
    :param n:
    :param P:
    :param Pr:
    :return:
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        mg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(mg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.amin(Pr) < 0 or np.amax(Pr) > 1:
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1])[0]:
        raise ValueError("Pr must sum to 1")
