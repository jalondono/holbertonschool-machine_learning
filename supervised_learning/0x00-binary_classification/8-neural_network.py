#!/usr/bin/env python3
""" Neural Network """
import numpy as np


class NeuralNetwork:
    """"neural network with one hidden layer performing
    binary classification:"""

    def __init__(self, nx, nodes):
        """
        Construtor
        :param nx: is the number of input features
        :param nodes: is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nodes < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        """The weights vector for the hidden layer."""
        self.W1 = np.random.normal(0, 1, (nodes, nx))
        """The bias for the hidden layer"""
        self.b1 = np.array([[0.], [0.], [0]])
        """The activated output for the hidden layer."""
        self.A1 = 0

        """The weights vector for the output neuron"""
        self.W2 = np.random.normal(0, 1, (1, 3))
        """The bias for the output neuron."""
        self.b2 = 0
        """The activated output for the output neuron (prediction"""
        self.A2 = 0
