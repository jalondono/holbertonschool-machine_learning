#!/usr/bin/env python3
""" Gradient Descent with Dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """

    :param Y:is a one-hot numpy.ndarray of shape (classes, m)
     that contains the correct labels for the data
    :param weights: is a dictionary of the weights
    and biases of the neural network
    :param cache: is a dictionary of the outputs and dropout masks
     of each layer of the neural network
    :param alpha: is the learning rate
    :param keep_prob: is the probability that a node will be kept
    :param L: is the number of layers of the network
    :return:
    """
    m = Y.shape[1]
    copy_weight = weights.copy()
    for idx in reversed(range(L)):
        current_A = cache['A' + str(idx + 1)]
        preview_A = cache['A' + str(idx)]

        if idx == L - 1:
            dz = current_A - Y
            dw = np.matmul(preview_A, dz.T) / m
        else:
            dz1a = np.matmul(copy_weight['W' + str(idx + 2)].T, dz)
            g_prime = 1 - current_A ** 2
            dz = dz1a * g_prime
            dz *= cache["D{}".format(idx + 1)]
            dz /= keep_prob
            dw = (np.matmul(preview_A, dz.T)) / m
        db = (np.sum(dz, axis=1, keepdims=True)) / m

        weights['W' + str(idx + 1)] = \
            (copy_weight['W' + str(idx + 1)] -
             (alpha * dw).T)

        weights['b' + str(idx + 1)] = \
            copy_weight['b' + str(idx + 1)] - (alpha * db)
