#!/usr/bin/env python3
"""The Forward Algorithm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model:
    :param Observation:
    :param Emission:
    :param Transition:
    :param Initial:
    :return:
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    init_observation = Observation[0]
    a = Initial.T
    b = Emission[:, init_observation]
    F[:, 0] = a * b
    for t in range(1, T):
        for j in range(Transition.shape[0]):
            X = F[:, t - 1]
            Y = Transition[:, j]
            Z = Emission[j, Observation[t]]
            F[j, t] = np.sum(X * Y * Z)
    P = np.sum(F[:, -1:])
    return P, F
