#!/usr/bin/env python3
""" Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    that conducts forward propagation using Dropout:
    :param X: is a numpy.ndarray of shape (nx, m) containing
     the input data for the network
    :param weights: is a dictionary of the weights and biases
     of the neural network
    :param L: the number of layers in the network
    :param keep_prob: is the probability that a node will be kept
    :return: a dictionary containing the outputs of each layer and
    the dropout mask used on each layer (see example for format)
    """
    cache = {'A0': X}
    for idx in range(L):
        curr_W = "W" + str(idx + 1)
        curr_A = "A" + str(idx)
        curr_B = "b" + str(idx + 1)
        next_A = "A" + str(idx + 1)
        curr_D = "D" + str(idx + 1)

        Z = np.matmul(weights[curr_W], cache[curr_A]) + weights[curr_B]
        drop = np.random.binomial(1, keep_prob, size=Z.shape)
        if idx != L - 1:
            cache[next_A] = np.tanh(Z)
            cache[curr_D] = drop
            cache[next_A] = (cache[next_A] * cache[curr_D]) / keep_prob
        else:
            t = np.exp(Z)
            cache[next_A] = t / np.sum(t, axis=0, keepdims=True)
    return cache
