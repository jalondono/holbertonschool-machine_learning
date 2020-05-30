#!/usr/bin/env python3
"""Early Stopping"""

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """

    :param cost: is the current validation cost of the neural network
    :param opt_cost: is the lowest recorded validation cost
     of the neural network
    :param threshold: is the threshold used for early stopping
    :param patience: is the patience count used for early stopping
    :param count: is the count of how long the threshold has not been met
    :return: a boolean of whether the network should be
    stopped early, followed by the updated count
    """
    current_err = abs(opt_cost - cost)
    if current_err > threshold:
        return False, 0
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
