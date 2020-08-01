#!/usr/bin/env python3
"""Likelihood"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given various
     hypothetical probabilities of developing severe side effects:
    :param x: is the number of patients that develop severe side effects
    :param n: is the total number of patients observed
    :param P: is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    :return:
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        mg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(mg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
