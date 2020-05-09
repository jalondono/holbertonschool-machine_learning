#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


def sigmoid(Z):
    """
    Sigmoid activation function
    :param Z: is the array of W.X + b values
    :return: Y predicted
    """
    sigma = (1.0 / (1.0 + np.exp(-Z)))
    return sigma


class DeepNeuralNetwork:
    """Deep neural network performing binary classification:"""

    def __init__(self, nx, layers):
        """
        Constructor
        :param nx: is the number of input features
        :param layers:  is a list representing the number
         of nodes in each layer of the network
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        layers_dims = [nx]
        layers_dims.extend(layers)
        L = len(layers_dims) - 1  # integer representing the number of layers
        for l in range(1, L + 1):
            if not isinstance(layers_dims[l], int) or layers_dims[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights['W' + str(l)] = \
                np.random.randn(layers_dims[l],
                                layers_dims[l - 1]) * \
                np.sqrt(2 / layers_dims[l - 1])
            self.__weights['b' + str(l)] = np.zeros((layers_dims[l], 1))

    @property
    def L(self):
        """Return L attribute"""
        return self.__L

    @property
    def cache(self):
        """Return cache attribute"""
        return self.__cache

    @property
    def weights(self):
        """Return weights attribute"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: that contains the input data
        :return:
        """
        last_idx = 0
        self.__cache['A0'] = X
        for idx, value in enumerate(self.layers):
            last_idx = idx + 1
            idx_layer = str(idx + 1)
            act_W = self.__weights['W' + idx_layer]
            act_b = self.__weights['b' + idx_layer]
            act_X = self.__cache['A' + str(idx)]
            self.__cache['A' + idx_layer] = \
                sigmoid(np.matmul(act_W, act_X) + act_b)
        return self.__cache['A' + str(last_idx)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: contains the correct labels for the input data
        :param A: containing the activated output of the neuron for each
        :return:
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) *
                        np.log(1.0000001 - A))) / m
        return cost
