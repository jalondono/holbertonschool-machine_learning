#!/usr/bin/env python3
""" Train keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent:
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
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
