#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification:"""

    def __init__(self, nx, layers):
        """ constructor """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

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
