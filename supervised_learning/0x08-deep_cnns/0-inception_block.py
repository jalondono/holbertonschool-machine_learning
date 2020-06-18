#!/usr/bin/env python3
"""Inception Block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in Going
    Deeper with Convolutions (2014):
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F1, F3R,
     F3,F5R, F5, FPP, respectively:
    :return:
    """

    initializer = K.initializers.he_normal(seed=None)

    # Layer 1
    l1 = K.layers.Conv2D(filters[0],
                         (1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer)(A_prev)

    l2 = K.layers.Conv2D(filters[1], (1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer)(A_prev)
    l2 = K.layers.Conv2D(filters[2], (3, 3),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer)(l2)

    l3 = K.layers.Conv2D(filters[3], (1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer)(A_prev)
    l3 = K.layers.Conv2D(filters[4], (5, 5),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer)(l3)

    l4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                               strides=(1, 1),
                               padding='same')(A_prev)
    l4 = K.layers.Conv2D(filters[5], (1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=initializer)(l4)
    out_1 = K.layers.concatenate([l1, l2, l3, l4])
    return out_1
