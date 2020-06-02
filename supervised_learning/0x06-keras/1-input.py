#!/usr/bin/env python3
""" Input keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:
    :param nx: is the number of input features to the network
    :param layers: is a list containing the number of nodes
    in each layer of the network
    :param activations: is a list containing the activation functions
    used for each layer of the network
    :param lambtha: is the L2 regularization parameter
    :param keep_prob: is the probability that a node will be kept for dropout
    :return: the keras model
    """
    reg = K.regularizers.l2(lambtha)
    input = K.Input(shape=(nx,))
    idx = 0
    past_layer = input
    for idx in range(len(layers) - 1):
        hidden = K.layers.Dense(layers[idx],
                                activation=activations[idx],
                                kernel_regularizer=reg)(past_layer)
        dropout = K.layers.Dropout(1 - keep_prob)(hidden)
        past_layer = dropout

    idx += 1
    output = K.layers.Dense(layers[idx],
                            activation=activations[idx],
                            kernel_regularizer=reg)(past_layer)
    model = K.models.Model(inputs=input, outputs=output)
    return model
