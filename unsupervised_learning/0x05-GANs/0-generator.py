#!/usr/bin/env python3
"""Generator"""
import tensorflow as tf
import numpy as np


def generator(Z):
    """
    creates a simple generator network for MNIST digits:
    :param Z: tf.tensor containing the input to the generator network
    :return: X, a tf.tensor containing the generated image
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.Dense(units=128, name='layer_1',
                                  activation=tf.nn.relu)(Z)

        layer_2 = tf.layers.Dense(units=784, name='layer_2',
                                  activation=tf.nn.sigmoid)

        X = layer_2(layer_1)
    return X
