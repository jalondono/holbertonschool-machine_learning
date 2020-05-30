#!/usr/bin/env python3
"""Create a Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a Layer with L2 Regularization
    :param prev: is a tensor containing the output of the previous layer
    :param n: is the number of nodes the new layer should contain
    :param activation: is the activation function that should be
     used on the layer
    :param lambtha: is the L2 regularization paramete
    :return: the output of the new layer
    """
    init_reg = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    model = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init_reg,
                            kernel_regularizer=kernel_regularizer)
    return model(prev)
