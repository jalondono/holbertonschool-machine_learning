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
    # kernel_init = K.initializers.he_normal(seed=None)
    # shape = X.shape
    #
    # model = K.Sequential()
    # model.add(K.layers.Conv2D(filters=6,
    #                           kernel_size=(5, 5),
    #                           padding='same',
    #                           activation='relu',
    #                           kernel_initializer=kernel_init,
    #                           input_shape=shape[1:]))
    #
    # model.add(K.layers.MaxPool2D(pool_size=(2, 2),
    #                              strides=(2, 2)))
    #
    # model.add(K.layers.Conv2D(filters=16,
    #                           kernel_size=(5, 5),
    #                           padding='valid',
    #                           activation='relu',
    #                           kernel_initializer=kernel_init))
    #
    # model.add(K.layers.MaxPool2D(pool_size=(2, 2),
    #                              strides=(2, 2)))
    #
    # model.add(K.layers.Flatten())
    #
    # model.add(K.layers.Dense(units=120,
    #                          activation='relu',
    #                          kernel_initializer=kernel_init,
    #                          ))
    #
    # model.add(K.layers.Dense(units=84,
    #                          activation='relu',
    #                          kernel_initializer=kernel_init,
    #                          ))
    #
    # model.add(K.layers.Dense(units=10,
    #                          kernel_initializer=kernel_init,
    #                          activation='softmax'
    #                          ))
    #
    # model.compile(optimizer=K.optimizers.Adam(),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # return model
    init = K.initializers.he_normal(seed=None)
    output = K.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=init,
                             activation='relu')(X)

    output2 = K.layers.MaxPool2D(strides=2)(output)

    output3 = K.layers.Conv2D(filters=16,
                              kernel_size=5,
                              padding='valid',
                              kernel_initializer=init,
                              activation='relu')(output2)

    output4 = K.layers.MaxPool2D(strides=2)(output3)

    output5 = K.layers.Flatten()(output4)

    output6 = K.layers.Dense(units=120,
                             kernel_initializer=init,
                             activation='relu')(output5)

    output7 = K.layers.Dense(units=84,
                             kernel_initializer=init,
                             activation='relu')(output6)

    output8 = K.layers.Dense(units=10,
                             kernel_initializer=init,
                             activation='softmax')(output7)

    model = K.models.Model(inputs=X, outputs=output8)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
