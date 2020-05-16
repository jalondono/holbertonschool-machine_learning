#!/usr/bin/env python3
"""Layers"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Layers
    :param prev:  is the tensor output of the previous layer
    :param n: s the number of nodes in the layer to create
    :param activation:  is the activation function that the layer should use
    :return:
    """
    initialization = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        kernel_initializer=initialization,
        activation=activation,
        name='layer')
    return layer
