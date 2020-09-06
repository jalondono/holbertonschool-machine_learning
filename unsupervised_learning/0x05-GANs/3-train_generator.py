#!/usr/bin/env python3
"""generator"""
import tensorflow as tf
import numpy as np
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """
    creates the loss tensor and training op for the generator:
    :param Z: is the tf.placeholder that is the input for the generator
    :return: loss, train_op
    """
    G_loss = -tf.reduce_mean(tf.log(Z))
    gen_vars = [var for var in tf.trainable_variables()
                if var.name.startswith("gen")]
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=gen_vars)

    return G_loss, G_solver
