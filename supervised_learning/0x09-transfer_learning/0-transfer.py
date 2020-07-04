#!/usr/bin/env python3
"""Transfer Knowledge """
import numpy as np
import scipy
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
import tensorflow.keras as K
from tensorflow.keras.applications import resnet50
import matplotlib.pyplot as plt

"""Importing de data training"""


def preprocess_data(X, Y):
    """ pre-processes the data for your model
        @X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
            where m is the number of data points
        @Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
        Returns: X_p, Y_p
            X_p: numpy.ndarray containing the preprocessed X
            Y_p: numpy.ndarray containing the preprocessed Y
    """

    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    """ trains the model and save it """
    batch_size = 32
    num_classes = 10
    epochs = 20

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    """Setting up the model"""

    original_dim = (32, 32, 3)
    target_size = (224, 224)

    resN = resnet50.ResNet50(include_top=False, weights='imagenet',
                             input_shape=original_dim,
                             pooling='max')
    # Defrost last layer of resnet50
    for layer in resN.layers[:-32]:
        layer.trainable = False

    # Concatenate the resnet with a modified top layer
    res_model = K.Sequential()
    res_model.add(K.layers.Lambda(lambda image:
                                  tf.image.resize(image, target_size)))
    res_model.add(resN)
    res_model.add(K.layers.Dense(512, activation='relu'))
    res_model.add(K.layers.Dropout(0.3))
    res_model.add(K.layers.Dense(512, activation='relu'))
    res_model.add(K.layers.Dropout(0.5))
    res_model.add(K.layers.Dense(10, activation='softmax'))

    opt = K.optimizers.RMSprop()
    # Let's train the model using RMSprop
    res_model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['acc'])

    lrr = K.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=.01,
        patience=3,
        min_lr=1e-5)

    es = K.callbacks.EarlyStopping(monitor='val_acc',
                                   mode='max',
                                   verbose=1,
                                   patience=10)

    mc = K.callbacks.ModelCheckpoint('cifar10.h5',
                                     monitor='val_acc',
                                     mode='max',
                                     verbose=1,
                                     save_best_only=True)

    res_model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  verbose=1,
                  callbacks=[mc, lrr, es]
                  )
    res_model.save('cifar10.h5')
