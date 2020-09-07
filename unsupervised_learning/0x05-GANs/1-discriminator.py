#!/usr/bin/env python3
"""Discriminator"""
import tensorflow as tf
import numpy as np


def discriminator(Z):
    """
    creates a simple generator network for MNIST digits:
    :param Z: tf.tensor containing the input to the generator network
    :return: X, a tf.tensor containing the generated image
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(Z, 128, activation=tf.nn.relu)
        x = tf.layers.dense(Z, 784)
        x = tf.nn.sigmoid(x)
    return x

