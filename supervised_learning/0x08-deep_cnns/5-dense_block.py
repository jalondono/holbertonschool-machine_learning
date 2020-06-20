#!/usr/bin/env python3
"""   Dense Block  """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected
     Convolutional Networks:
    :param X:
    :param nb_filters:
    :param growth_rate:
    :param layers:
    :return:
    """
    initializer = K.initializers.he_normal(seed=None)

    for i in range(layers):
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)

        # conv 1×1 produces 4k (growth_rate) feature-maps
        layer = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                )(layer)

        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)

        # conv 3×3 produces k (growth_rate) feature-maps
        layer = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=initializer,
                                )(layer)

        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate

    return X, nb_filters
