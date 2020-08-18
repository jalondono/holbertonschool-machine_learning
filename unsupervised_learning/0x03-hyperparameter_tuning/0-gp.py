#!/usr/bin/env python3
"""Initialize Gaussian Process """
import numpy as np


class GaussianProcess:
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Create the class GaussianProcess that represents a
        noiseless 1D Gaussian process:
        :param X_init: is a numpy.ndarray of shape (t, 1)
         representing the inputs already sampled with the black-box function
        :param Y_init: is a numpy.ndarray of shape (t, 1) representing the
         outputs of the black-box function for each input in X_init
        :param l: is the length parameter for the kernel
        :param sigma_f: is the standard deviation given to the output of
         the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.sigma_f = sigma_f
        self.l = l
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """

        :param X1: is a numpy.ndarray of shape (m, 1)
        :param X2: is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        :return: the covariance kernel matrix as a numpy.ndarray
         of shape (m, n)
        """
        a = np.sum(X1 ** 2, 1).reshape(-1, 1)
        b = np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        sqdist = a + b
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
