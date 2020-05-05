#!/usr/bin/env python3
""" Neuron """
import numpy as np


class Neuron:
    """Neuron Class"""

    def __init__(self, nx):
        """constructor method"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0
