#!/usr/bin/env python3
""" DenseNet-121 """
import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture
    :param growth_rate: growth rate
    :param compression: compression factor
    :return: keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]

    layer = K.layers.BatchNormalization(axis=3)(X)
    layer = K.layers.Activation('relu')(layer)

    layer = K.layers.Conv2D(filters=2 * growth_rate,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer,
                            )(layer)

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               padding='same',
                               strides=(2, 2))(layer)

    nb_filters = 2 * growth_rate

    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 6)

    for l in layers:
        layer, nb_filters = transition_layer(layer,
                                             nb_filters,
                                             compression)

        layer, nb_filters = dense_block(layer,
                                        nb_filters,
                                        growth_rate,
                                        l)

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=initializer,
                           )(layer)

    model = K.models.Model(inputs=X, outputs=layer)

    return model
