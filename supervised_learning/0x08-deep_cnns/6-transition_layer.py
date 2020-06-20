#!/usr/bin/env python3
"""contains the transition_layer function"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer
    :param X: output from the previous layer
    :param nb_filters: integer representing the number of filters in X
    :param compression: compression factor for the transition layer
    :return: output of the transition layer and
        the number of filters within the output
    """

    init = K.initializers.he_normal(seed=None)

    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)

    nb_filters = int(nb_filters * compression)

    layer = K.layers.Conv2D(filters=nb_filters,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=init,
                            )(layer)

    X = K.layers.AveragePooling2D(pool_size=2,
                                  padding='same')(layer)

    return X, nb_filters
