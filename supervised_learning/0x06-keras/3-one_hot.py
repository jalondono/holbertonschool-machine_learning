#!/usr/bin/env python3
""" Optimize keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    One Hot
    :param labels:
    :param classes:
    :return:
    """
    return K.utils.to_categorical(
        labels,
        num_classes=classes
    )
