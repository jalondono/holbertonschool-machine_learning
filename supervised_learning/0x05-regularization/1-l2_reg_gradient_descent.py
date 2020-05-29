#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient
     descent with L2 regularization:
    :param Y:
    :param weights:
    :param cache:
    :param alpha:
    :param lambtha:
    :param L:
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
            dw = (np.matmul(preview_A, dz.T)) / m
        db = (np.sum(dz, axis=1, keepdims=True)) / m

        weights['W' + str(idx + 1)] = \
            (copy_weight['W' + str(idx + 1)] -
             ((alpha * lambtha / m) * (copy_weight['W' + str(idx + 1)])) -
             (alpha * dw).T)

        weights['b' + str(idx + 1)] = \
            copy_weight['b' + str(idx + 1)] - (alpha * db)
