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
    predict_ones = tf.argmax(y_pred, 1)
    label_ones = tf.argmax(y, 1)

    equals = tf.math.equal(predict_ones, label_ones)
    mean = tf.reduce_mean(tf.cast(equals, tf.float32), axis=0)
    # ptional way
    # mean = tf.metrics.accuracy(y, y_pred)[0]
    return mean
