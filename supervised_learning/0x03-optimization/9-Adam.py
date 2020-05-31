#!/usr/bin/env python3
"""Adam optimization"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates a variable in place using the Adam optimization algorithm:"""
    v = (v * beta1) + ((1 - beta1) * grad)
    s = (s * beta2) + ((1 - beta2) * (grad ** 2))
    v_corrected = v / (1 - (beta1 ** t))
    s_corrected = s / (1 - (beta2 ** t))
    var = var - ((alpha * v_corrected) / (s_corrected ** (1 / 2) + epsilon))
    return var, v, s
