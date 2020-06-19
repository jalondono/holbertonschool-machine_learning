#!/usr/bin/env python3
"""  Identity Bloc"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
     builds an identity block as described in Deep Residual
     Learning for Image Recognition (2015):
    :param A_prev:
    :param filters:
    :return:
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    layer = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init,
                            )(A_prev)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)

    layer = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=init)(layer)

    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)

    layer = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            kernel_initializer=init,
                            padding='same')(layer)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)

    output = K.layers.Add()([layer, A_prev])
    output = K.layers.Activation('relu')(output)

    return output
