#!/usr/bin/env python3
""" Neural Network """
import numpy as np


def sigmoid(Z):
    """
    Sigmoid activation function
    :param Z: expression
    :return:
    """
    sigma = (1.0 / (1.0 + np.exp(-Z)))
    return sigma


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
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        """The weights vector for the hidden layer."""
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        """The bias for the hidden layer"""
        self.__b1 = np.zeros((nodes, 1))
        """The activated output for the hidden layer."""
        self.__A1 = 0

        """The weights vector for the output neuron"""
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        """The bias for the output neuron."""
        self.__b2 = 0
        """The activated output for the output neuron (prediction"""
        self.__A2 = 0

    @property
    def W1(self):
        """return W1 Value"""
        return self.__W1

    @property
    def b1(self):
        """return b1 Value"""
        return self.__b1

    @property
    def A1(self):
        """return A1 Value"""
        return self.__A1

    @property
    def W2(self):
        """return W2 Value"""
        return self.__W2

    @property
    def b2(self):
        """return b2 Value"""
        return self.__b2

    @property
    def A2(self):
        """return A2 Value"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation
         of the neural network
        :param X: contains the input data
        :return: the private attributes __A1 and __A2
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """

        :param Y: contains the correct labels for the input data
        :param A: containing the activated output of the neuron
        for each example
        :return: the cost
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) *
                        np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """

        :param X: contains the input data
        :param Y: contains the correct labels for the input data
        :return:
        """

        A = self.forward_prop(X)[1]
        cost = self.cost(Y, A)
        A_evaluate = np.where(A >= 0.5, 1, 0)
        return A_evaluate, cost
