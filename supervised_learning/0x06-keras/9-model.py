#!/usr/bin/env python3
"""  Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model:
    :param network: is the model to save
    :param filename: is the path of the file that the
     model should be saved to
    :return: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loads an entire model:
    :param filename: is the path of the file that the
     model should be loaded from
    :return: the loaded model
    """
    return K.models.load_model(filepath=filename)
