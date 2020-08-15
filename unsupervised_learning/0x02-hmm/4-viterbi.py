#!/usr/bin/env python3
""" The Viretbi Algorithm """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
     calculates the most likely sequence of hidden
     states for a hidden markov model:
    :param Observation:
    :param Emission:
    :param Transition:
    :param Initial:
    :return:
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

        # dim conditions
    T = Observation.shape[0]

    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # stochastic
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    # initialization
    Obs_t = Observation[0]
    backpointer[:, 0] = 0
    prob = np.multiply(Initial[:, 0], Emission[:, Obs_t])
    viterbi[:, 0] = prob

    # recursion
    for t in range(1, T):
        a = viterbi[:, t - 1]
        b = Transition.T
        ab = a * b
        ab_max = np.amax(ab, axis=1)
        c = Emission[:, Observation[t]]
        prob = ab_max * c

        viterbi[:, t] = prob
        backpointer[:, t - 1] = np.argmax(ab, axis=1)

    # path initialization
    path = []
    current = np.argmax(viterbi[:, T - 1])
    path = [current] + path

    # path backwards traversing
    for t in range(T - 2, -1, -1):
        current = int(backpointer[current, t])
        path = [current] + path

    # max path probabilities among all possible states
    # end of path

    P = np.amax(viterbi[:, T - 1], axis=0)

    return path, P
