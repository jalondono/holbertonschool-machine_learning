#!/usr/bin/env python3
""" Neuron """
import numpy as np


def sigmoid(Z):
    """Activation function of sigmoid neurone"""
    sigma = (1.0 / (1.0 + np.exp(-Z)))
    return sigma


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
        self.__W = np.random.normal(0, 1, (1, nx))

        """The bias for the neuron. Upon instantiation,
         it should be initialized to 0."""
        self.__b = 0

        """The activated output of the neuron (prediction).
         Upon instantiation, it should be initialized to 0."""
        self.__A = 0

    @property
    def W(self):
        """Private instance of W"""
        return self.__W

    @property
    def b(self):
        """Private instance of b"""
        return self.__b

    @property
    def A(self):
        """Private instance of A"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        :param X: is a numpy.ndarray with shape (nx, m)
         that contains the input data
        :return: The private attribute __A
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: Contains the correct labels for the input data
        :param A: containing the activated output of the neuron for each
        :return: Cost
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum((Y * np.log(A) + ((1.0000001 - A) * np.log(1.0000001 - A))))
        return cost
