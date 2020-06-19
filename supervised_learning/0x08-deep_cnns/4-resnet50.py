#!/usr/bin/env python3
"""  ResNet-50 """
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in Deep Residual
     Learning for Image Recognition (2015):
    :return:
    """
    init = K.initializers.he_normal(seed=None)
    data_in = K.Input(shape=(224, 224, 3))
    layer = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=init
                            )(data_in)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation(activation='relu')(layer)

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer = projection_block(layer, [64, 64, 256], 1)
    for i in range(2):
        layer = identity_block(layer, [64, 64, 256])

    layer = projection_block(layer, [128, 128, 512])
    for i in range(3):
        layer = identity_block(layer, [128, 128, 512])

    layer = projection_block(layer, [256, 256, 1024])
    for i in range(5):
        layer = identity_block(layer, [256, 256, 1024])

    layer = projection_block(layer, [512, 512, 2048])
    for i in range(2):
        layer = identity_block(layer, [512, 512, 2048])

    layer = K.layers.AvgPool2D(pool_size=(7, 7),
                               padding='same')(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=init,
                           )(layer)

    model = K.models.Model(inputs=data_in, outputs=layer)

    return model
