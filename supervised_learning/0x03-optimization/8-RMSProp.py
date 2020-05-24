#!/usr/bin/env python3
""" RMSProp Upgraded"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """creates the training operation for a neural network
     in tensorflow using the RMSProp optimization algorithm:"""
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        epsilon=epsilon,
        decay=beta2)
    train = optimizer.minimize(loss)
    return train
