#!/usr/bin/env python3
""" Epsilon Greedy"""
import numpy


def epsilon_greedy(Q, state, epsilon):
    """
    uses epsilon-greedy to determine the next action:
    :param Q: is a numpy.ndarray containing the q-table
    :param state: is the current state
    :param epsilon: is the epsilon to use for the calculation
    :return: the next action index
    """
    p = numpy.random.uniform(0, 1)
    if p < epsilon:
        # explore
        action = numpy.random.randint(Q.shape[1])
    else:
        action = numpy.argmax(Q[state, :])
    return action
