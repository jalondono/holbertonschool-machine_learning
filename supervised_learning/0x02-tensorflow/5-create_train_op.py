#!/usr/bin/env python3
""" Train_Op """
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network:
    :param loss: is the loss of the network’s prediction
    :param alpha: is the learning rate
    :return: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
