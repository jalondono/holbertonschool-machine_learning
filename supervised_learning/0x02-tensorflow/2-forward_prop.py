#!/usr/bin/env python3
"""Forward Propagation """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network:
    :param x: Is the placeholder for the input data
    :param layer_sizes: Is a list containing the number
     of nodes in each layer of the network
    :param activations:  is a list containing the
     activation functions for each layer of the network
    :return: the prediction of the network in tensor form
    """

    predicted_nn = create_layer(x, layer_sizes[0], activations[0])
    for idx in range(1, len(layer_sizes)):
        predicted_nn = create_layer(predicted_nn, layer_sizes[idx], activations[idx])
    return predicted_nn

