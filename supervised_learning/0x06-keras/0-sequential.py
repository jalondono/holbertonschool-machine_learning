#!/usr/bin/env python3
""" Sequential keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:
    :param nx: is the number of input features to the network
    :param layers: is a list containing the number of nodes in each layer
     of the network
    :param activations: is a list containing the activation functions used
    for each layer of the network
    :param lambtha: is the L2 regularization parameter
    :param keep_prob: is the probability that a node will be kept for dropout
    :return: the keras model
    """

    reg = K.regularizers.l2(lambtha)
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_regularizer=reg,
                             input_shape=(nx,),
                             ))

    for idx in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(units=layers[idx],
                                 activation=activations[idx],
                                 kernel_regularizer=reg
                                 ))
    return model
