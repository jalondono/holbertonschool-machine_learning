#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization:
    :param cost:is the cost of the network without L2 regularization
    :param lambtha: is the regularization parameter
    :param weights: is a dictionary of the weights and biases
     (numpy.ndarrays) of the neural network
    :param L: is the number of layers in the neural network
    :param m: is the number of data points used
    :return:the cost of the network accounting for L2 regularization
    """
    sumatory = 0
    for key, value in weights.items():
        aux = np.sum(value ** 2) ** 0.5
        sumatory += aux
    l2 = (lambtha / (2 * m)) * sumatory
    l2_reg = cost + l2
    return l2_reg
