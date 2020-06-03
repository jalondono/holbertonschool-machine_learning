#!/usr/bin/env python3
"""  Predict Keras  """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network:
    :param network:
    :param data:
    :param verbose:
    :return:
    """
    pred = network.predict(data, verbose=verbose)
    return pred
