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