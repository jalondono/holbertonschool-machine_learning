#!/usr/bin/env python3
"""L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization:
    :param cost: is a tensor containing the cost of the network
    without L2 regularization
    :return: : a tensor containing the cost of the network accounting
     for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
