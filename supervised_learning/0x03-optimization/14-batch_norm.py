#!/usr/bin/env python3
""" Batch Normalization Upgraded """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer
    for a neural network in tensorflow:"""

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(units=n,
                            activation=None,
                            kernel_initializer=init,
                            name='layer')
    mean, variance = tf.nn.moments(layer(prev), axes=0, keepdims=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)

    bn = tf.nn.batch_normalization(layer(prev),
                                   mean=mean,
                                   variance=variance,
                                   offset=beta,
                                   scale=gamma,
                                   variance_epsilon=1e-8)
    if activation is None:
        return layer(prev)
    return activation(bn)
