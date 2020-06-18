#!/usr/bin/env python3
""" Inception Network"""
import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
     builds the inception network as described
      in Going Deeper with Convolutions (2014):
    :return:
    """
    init = K.initializers.he_normal(seed=None)
    input_img = K.Input((224, 224, 3))

    init_layer = K.layers.Conv2D(filters=64,
                                 kernel_size=(7, 7),
                                 kernel_initializer=init,
                                 strides=(2, 2),
                                 padding='same',
                                 activation='relu')(input_img)

    init_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(init_layer)

    init_layer = K.layers.Conv2D(filters=64,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding='same',
                                 activation='relu',
                                 kernel_initializer=init,
                                 )(init_layer)

    init_layer = K.layers.Conv2D(filters=192,
                                 kernel_size=(3, 3),
                                 kernel_initializer=init,
                                 strides=(1, 1),
                                 padding='same',
                                 activation='relu')(init_layer)

    init_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(init_layer)

    init_layer = inception_block(init_layer, [64, 96, 128, 16, 32, 32])
    init_layer = inception_block(init_layer, [128, 128, 192, 32, 96, 64])

    init_layer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       padding='same',
                                       strides=(2, 2))(init_layer)

    init_layer = inception_block(init_layer, [192, 96, 208, 16, 48, 64])
    init_layer = inception_block(init_layer, [160, 112, 224, 24, 64, 64])
    init_layer = inception_block(init_layer, [128, 128, 256, 24, 64, 64])
    init_layer = inception_block(init_layer, [112, 144, 288, 32, 64, 64])
    init_layer = inception_block(init_layer, [256, 160, 320, 32, 128, 128])

    init_layer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same')(init_layer)
    init_layer = inception_block(init_layer, [256, 160, 320, 32, 128, 128])
    init_layer = inception_block(init_layer, [384, 192, 384, 48, 128, 128])
    init_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                           strides=(1, 1),
                                           padding='same')(init_layer)
    init_layer = K.layers.Dropout(rate=0.40)(init_layer)
    init_layer = K.layers.Dense(units=1000,
                                activation='softmax',
                                kernel_initializer=init)(init_layer)
    model = K.models.Model(inputs=input_img, outputs=init_layer)

    return model
