#!/usr/bin/env python3
""" The Backward Algorithm """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """  performs the backward algorithm for a hidden markov model:
        - Observation is a numpy.ndarray of shape (T,) that contains
            the index of the observation
            - T is the number of observations
        - Emission is a numpy.ndarray of shape (N, M) containing the
            emission probability of a specific observation given a
            hidden state
            - Emission[i, j] is the probability of observing j given
                the hidden state i
        - N is the number of hidden states
        - M is the number of all possible observations
        - Transition is a 2D numpy.ndarray of shape (N, N) containing
            the transition probabilities
            - Transition[i, j] is the probability of transitioning
                from the hidden state i to j
        - Initial a numpy.ndarray of shape (N, 1) containing the
            probability of starting in a particular hidden state
    Returns: P, B, or None, None on failure
        - P is the likelihood of the observations given the model
        - B is a numpy.ndarray of shape (N, T) containing the backward
            path probabilities
            - B[i, j] is the probability of being in hidden state
            i at time j
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    B = np.zeros((N, T))

    B[:, T - 1] = np.ones((N))

    for t in range(T - 2, -1, -1):
        for j in range(N):
            B[j, t] = (np.sum(B[:, t + 1] * Emission[:, Observation[t + 1]] *
                       Transition[j, :]))
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
