#!/usr/bin/env python3
""" Train keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1):
    """
    trains a model using mini-batch gradient descent:
    :param decay_rate:
    :param alpha:
    :param learning_rate_decay:
    :param patience:
    :param early_stopping:
    :param validation_data:
    :param network: is the model to train
    :param data: is a numpy.ndarray of shape (m, nx)
     containing the input data
    :param labels: is a one-hot numpy.ndarray of shape (m, classes)
     containing the labels of data
    :param batch_size: is the size of the batch used
     for mini-batch gradient descent
    :param epochs: is the number of passes through
     data for mini-batch gradient descent
    :param verbose: is a boolean that determines
     if output should be printed during training
    :param shuffle: is a boolean that determines
    whether to shuffle the batches every epoch.
     Normally, it is a
    :return: the History object generated after training the model
    """

    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)

    cb = []
    if validation_data and learning_rate_decay:
        cb.append(K.callbacks.LearningRateScheduler(learning_rate,
                                                    verbose=1))

    if validation_data and early_stopping:
        cb.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                            mode='min',
                                            patience=patience))
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=cb)
    return history
