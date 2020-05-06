#!/usr/bin/env python3
""" Neuron """
import numpy as np


class Neuron:
    """Neuron Class"""

    def __init__(self, nx):
        """
        constructor method
        nx is the number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        """
        W = The weights vector for the neuron. Upon instantiation
        using a random normal distribution.
        """
        self.W = np.random.normal(0, 1, (1, nx))

        """The bias for the neuron. Upon instantiation, it should be initialized to 0."""
        self.b = 0

        """The activated output of the neuron (prediction).
         Upon instantiation, it should be initialized to 0."""
        self.A = 0
