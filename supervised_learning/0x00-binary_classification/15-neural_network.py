#!/usr/bin/env python3
""" Neural Network """
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    """
    Sigmoid activation function
    :param Z: expression
    :return:
    """
    sigma = (1.0 / (1.0 + np.exp(-Z)))
    return sigma


def plot_training(data, iterations, step):
    """
    function to plot data training on matplotlib

    Plot the training data every step iterations as a blue line
    Include data from the 0th and last iteration

    :param data: list of cost vs iteration
    :param iterations: Number of iteration or step to the plot
    :return:
    """
    plt.xlim(-step, iterations)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Training Cost')
    plt.plot(data[0], data[1], 'b-')


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural n
        :param X: Contains the input data
        :param Y: Contains the correct labels for the input data
        :param A1: is the output of the hidden layer
        :param A2: is the predicted output
        :param alpha: is the learning rate
        :return: Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = A1.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        g_prime = A1 * (1 - A1)
        dz1a = np.matmul(self.__W2.T, dz2)
        dz1 = dz1a * g_prime
        dw1 = np.matmul(X, dz1.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - (alpha * dw2).T
        self.__b2 = self.__b2 - (alpha * db2)

        self.__W1 = self.__W1 - (alpha * dw1).T
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """

        :param X: contains the input data
        :param Y: contains the correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        :return: the evaluation of the training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if (verbose or graph) and not isinstance(step, int):
            raise TypeError('step must be an integer')
        if (verbose or graph) and step <= 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')
        data = [[], []]
        step_aux = 0
        for idx in range(iterations + 1):
            self.forward_prop(X)
            if (verbose or graph) and idx == 0:
                data[0].append(0)
                data[1].append(self.cost(Y, self.__A2))
                cost = self.cost(Y, self.__A2)
                step_aux += 1
                print('Cost after {} iterations: {}'.format(idx, cost))

            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose and (idx == (step * step_aux)):
                step_aux += 1
                cost = self.cost(Y, self.__A2)
                data[0].append(idx)
                data[1].append(cost)
                print('Cost after {} iterations: {}'.format(idx, cost))
        if graph:
            plot_training(data, iterations, step)
        return self.evaluate(X, Y)
