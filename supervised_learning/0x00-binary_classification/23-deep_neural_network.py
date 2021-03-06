#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    """
    Sigmoid activation function
    :param Z: is the array of W.X + b values
    :return: Y predicted
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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        :param X: contains the input data
        :param Y: contains the correct labels for the input data
        :return: neuron’s prediction and the cost of the network
        """
        self.forward_prop(X)
        aux_Y_evaluate = self.__cache['A' + str(len(self.__cache) - 1)]
        Y_evalueate = np.where(aux_Y_evaluate > 0.5, 1, 0)
        return Y_evalueate, self.cost(Y, aux_Y_evaluate)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on
         the neural network
        :param Y: contains the correct labels for the input data
        :param cache: s a dictionary containing all the
         intermediary values of the network
        :param alpha: is the learning rate
        :return: Updates the private attribute __weights
        """
        weights = self.__weights.copy()
        m = Y.shape[1]
        for idx in reversed(range(self.__L)):
            current_A = cache['A' + str(idx + 1)]
            preview_A = cache['A' + str(idx)]
            current_W = self.__weights['W' + str(idx + 1)]
            current_b = self.__weights['b' + str(idx + 1)]

            if idx == self.__L - 1:
                dz = current_A - Y
                dw = np.matmul(preview_A, dz.T) / m
            else:
                dz1a = np.matmul(weights['W' + str(idx + 2)].T, dz)
                g_prime = current_A * (1 - current_A)
                dz = dz1a * g_prime
                dw = (np.matmul(preview_A, dz.T)) / m
            db = (np.sum(dz, axis=1, keepdims=True)) / m

            self.__weights['W' + str(idx + 1)] = \
                (weights['W' + str(idx + 1)] - (alpha * dw).T)

            self.__weights['b' + str(idx + 1)] = \
                weights['b' + str(idx + 1)] - (alpha * db)

    def train(self, X, Y, iterations=100, alpha=0.05,
              verbose=True, graph=True, step=10):
        """
        rains the deep neural network by updating
        the private attributes
        :param X: contains the input data
        :param Y:  contains the correct labels for the input data
        :param iterations:  is the number of iterations to train ove
        :param alpha: is the learning rate
        :param verbose: is a boolean that defines whether or not to print
        :param graph: is a boolean that defines whether or not to graph
        :param step: step to print or graph
        :return: the evaluation of the training data
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        data = [[], []]
        step_aux = 0
        for idx in range(iterations + 1):
            self.forward_prop(X)
            if (verbose or graph) and idx == 0:
                data[0].append(0)
                data[1].append(self.cost(Y,
                                         self.__cache['A{}'.format(self.L)]))
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                step_aux += 1
                print('Cost after {} iterations: {}'.format(idx, cost))

            self.gradient_descent(Y, self.cache, alpha)
            if verbose and (idx == (step * step_aux)):
                step_aux += 1
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                data[0].append(idx)
                data[1].append(cost)
                print('Cost after {} iterations: {}'.format(idx, cost))
        if graph:
            plot_training(data, iterations, step)
        return self.evaluate(X, Y)
