#!/usr/bin/env python3
""" Neuron """
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    """Activation function of sigmoid neurone"""
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
    plt.xlabel('iteration)', fontsize='x-small')
    plt.ylabel('cost', fontsize='x-small')
    plt.title('Training Cost', fontsize='x-small')
    plt.plot(data[0], data[1], 'b-')


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
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) *
                        np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        :param X: that contains the input data
        :param Y: contains the correct labels for the input data
        :return: the neuron’s prediction and the cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_predict = np.where(A >= 0.5, 1, 0)
        return Y_predict, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        :param X: contains the input data
        :param Y: contains the correct labels for the input data
        :param A: containing the activated output of the neuron
        for each example
        :param alpha: is the learning rate
        :return: Just Updates the private attributes __W and __b
        """
        dz = A - Y
        dw = np.matmul(X, dz.T) / dz.shape[1]
        db = np.sum(dz) / dz.shape[1]

        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron
        :param X: contains the input data
        :param Y: contains the correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        :return:  the evaluation of the training data after iterations
        of training have occurred
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
                data[1].append(self.cost(Y, self.__A))
                cost = self.cost(Y, self.__A)
                step_aux += 1
                print('Cost after {} iterations: {}'.format(idx, cost))

            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose and (idx == (step * step_aux)):
                step_aux += 1
                cost = self.cost(Y, self.__A)
                data[0].append(idx)
                data[1].append(cost)
                print('Cost after {} iterations: {}'.format(idx, cost))
        if graph:
            plot_training(data, iterations, step)
        return self.evaluate(X, Y)
