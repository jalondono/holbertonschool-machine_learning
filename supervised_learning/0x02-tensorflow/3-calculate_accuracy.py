#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction:
    :param y: is a placeholder for the labels of the input data
    :param y_pred: is a tensor containing the network’s predictions
    :return: a tensor containing the decimal accuracy of the prediction
    """
    return tf.metrics.accuracy(y, y_pred)[0]
