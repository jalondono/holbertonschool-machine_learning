#!/usr/bin/env python3
"""Discriminator"""
import tensorflow as tf
import numpy as np
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """

    :param Z: is the tf.placeholder that is the input for
    the generator
    :param X: is the tf.placeholder that is the real input for
     the discriminator
    :return: loss, train_op
    """
    D_real = discriminator(X)
    D_fake = discriminator(Z)

    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))

    # Optimizers
    disc_vars = [var for var in tf.trainable_variables()
                 if var.name.startswith("disc")]
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=disc_vars)
    return D_loss, D_solver
