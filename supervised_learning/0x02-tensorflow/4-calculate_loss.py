#!/usr/bin/env python3
""" Loss """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction:
    :param y: is a placeholder for the labels of the input data
    :param y_pred: is a tensor containing the network’s predictions
    :return:
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)