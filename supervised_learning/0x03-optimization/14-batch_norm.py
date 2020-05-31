#!/usr/bin/env python3
""" Batch Normalization Upgraded """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer
    for a neural network in tensorflow:"""

    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n,
                            activation=None,
                            kernel_initializer=initializer,
                            name='layer')

    mean, variance = tf.nn.moments(model(prev), axes=0, keep_dims=True)

    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)

    adjusted = tf.nn.batch_normalization(model(prev), mean, variance,
                                         offset=beta, scale=gamma,
                                         variance_epsilon=1e-8)

    if activation is None:
        return model(prev)
    return activation(adjusted)
