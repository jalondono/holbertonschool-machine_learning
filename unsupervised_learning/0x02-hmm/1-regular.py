#!/usr/bin/env python3
"""Regular Chains """
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular
     markov chain:
    :param P: is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
    :return: a numpy.ndarray of shape (1, n) containing the steady state
     probabilities, or None on failure
    """

    if not isinstance(P, np.ndarray) or (P.shape[0] != P.shape[1]):
        return None

    idx = 0
    n = P.shape[0]
    p_aux = np.copy(P)
    p_prev = np.copy(P)
    p_list = [P]
    while True:
        p_aux = np.matmul(p_prev, p_aux)
        for item in p_list:
            if (item == p_aux).all() or idx == 1000:
                return None
        if (P > 0).sum() == P.size:
            break
        p_list.append(p_aux)
        idx += 1
    S = np.zeros(n)
    I = np.identity(n)

    # S(P-I) = 0
    # (P-I).T * S.T
    P_I = (P-I).T
    P_I[-1] = np.ones(n)
    S[-1] = 1
    result = np.linalg.solve(P_I, S).reshape(1, n)
    return result
