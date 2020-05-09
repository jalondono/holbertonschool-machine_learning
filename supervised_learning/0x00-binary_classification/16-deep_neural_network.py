#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification:"""

    def __init__(self, nx, layers):
        """ constructor """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if np.product(layers) <= 0:
            raise ValueError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        layers_dims = [nx]
        layers_dims.extend(layers)
        L = len(layers_dims) - 1  # integer representing the number of layers
        for l in range(1, L + 1):
            self.weights['W' + str(l)] = \
                np.random.randn(layers_dims[l],
                                layers_dims[l - 1]) * \
                np.sqrt(2 / layers_dims[l - 1])
            self.weights['b' + str(l)] = np.zeros((layers_dims[l], 1))
