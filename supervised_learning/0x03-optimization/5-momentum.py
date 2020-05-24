#!/usr/bin/env python3
"""Momentun"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum
    optimization algorithm:
    :param alpha: is the learning rate
    :param beta1: is the momentum weight
    :param var: is a numpy.ndarray containing the variable to be updated
    :param grad: is a numpy.ndarray containing the gradient of var
    :param v: is the previous first moment of var
    :return: the updated variable and the new moment, respectivel
    """
    v = v * beta1 + ((1 - beta1) * grad)
    var = var - (alpha * v)
    return var, v
