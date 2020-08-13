#!/usr/bin/env python3
""" Absorbing Chains """
import numpy as np


def check_connection(P, diag):
    """
    Check if the each states can connect to the absorbing state
    :param P: is the probability of transitioning from
     state i to state j
    :param diag: idx of each diagonal absorb
    :return:
    """
    n = P.shape[0]
    posible_states = diag.tolist()
    for i in range(2):
        for idx, state in enumerate(P):
            for absorb in posible_states:
                if state[absorb] != 0 and idx not in posible_states:
                    posible_states.append(idx)
                    break
    if len(posible_states) != n:
        return False
    return True


def absorbing(P):
    """
     determines if a markov chain is absorbing
    :param P: is the probability of transitioning from
     state i to state j
    :return: is the number of states in the markov chain
    """
    if not isinstance(P, np.ndarray) or (P.shape[0] != P.shape[1]):
        return False

    n = P.shape[0]
    diag_aux = np.ones(n)
    diag = np.diag(P)

    if (diag == diag_aux).all():
        return True
    if (diag == 1).sum() < 1:
        return False
    idx_absor = np.where(diag == 1)[0]
    return check_connection(P, idx_absor)
