#!/usr/bin/env python3
"""Markov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain being in a particular
     state after a specified number of iterations:
    :param P:  is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
    :param s: is a numpy.ndarray of shape (1, n) representing the
     probability of starting in each state
    :param t:  is the number of iterations that the markov chain has
    been through
    :return: a numpy.ndarray of shape (1, n) representing the
     probability of being in a specific state after t iterations,
      or None on failure
    """
    if not isinstance(t, int) or t <= 0:
        return None

    # P edge cases
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    # s edge cases
    if not isinstance(s, np.ndarray) or s.shape[1] != P.shape[0]:
        return None

    prob_state = np.copy(s)
    for idx in range(t):
        prob_state = np.matmul(prob_state, P)
    return prob_state
