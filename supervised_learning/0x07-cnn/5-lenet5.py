#!/usr/bin/env python3
"""LeNet-5 (Keras"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5
    architecture using keras:
    :param X: is a K.Input of shape (m, 28, 28, 1)
     containing the input images for the network
    :return: a K.Model compiled to use Adam optimization
     (with default hyperparameters) and accuracy metrics
    """
    kernel_init = K.initializers.he_normal(seed=None)
    shape = X.shape

    model = K.Sequential()
    model.add(K.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu',
                              kernel_initializer=kernel_init,
                              input_shape=shape[1:]))

    model.add(K.layers.MaxPool2D(input_shape=(2, 2),
                                 strides=(2, 2)))

    model.add(K.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu',
                              kernel_initializer=kernel_init))

    model.add(K.layers.MaxPool2D(input_shape=(2, 2),
                                 strides=(2, 2)))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(units=120,
                             activation='relu',
                             kernel_initializer=kernel_init,
                             ))

    model.add(K.layers.Dense(units=84,
                             activation='relu',
                             kernel_initializer=kernel_init,
                             ))

    model.add(K.layers.Dense(units=10,
                             kernel_initializer=kernel_init,
                             activation='softmax'
                             ))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
