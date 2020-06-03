#!/usr/bin/env python3
"""  Test Keras  """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network:
    :param network: is the network model to test
    :param data: is the input data to test the model with
    :param labels: are the correct one-hot labels of data
    :param verbose: is a boolean that determines if output
     should be printed during the testing process
    :return: the loss and accuracy of the model with the testing
     data, respectively
    """
    results = network.evaluate(x=data,
                               y=labels,
                               verbose=verbose)
    return results
