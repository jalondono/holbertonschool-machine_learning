#!/usr/bin/env python3
"""  Projection Block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
     builds a projection block as described in Deep Residual
      Learning for Image Recognition (2015):
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F11, F3, F12, respectively
    :param s: is the stride of the first convolution in both the main
     path and the shortcut connection
    :return:
    """
    F11, F3, F12 = filters

    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               strides=(s, s)
                               )(A_prev)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # Conv 3x3
    my_layer = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)

    extra_layer = K.layers.Conv2D(filters=F12,
                                  kernel_size=(1, 1),
                                  padding='same',
                                  kernel_initializer=initializer,
                                  strides=(s, s)
                                  )(A_prev)

    extra_layer = K.layers.BatchNormalization(axis=3)(extra_layer)

    output = K.layers.Add()([my_layer, extra_layer])

    output = K.layers.Activation('relu')(output)

    return output
