#!/usr/bin/env python3
"""Initialize Gaussian Process """
import numpy as np


class GaussianProcess:
    """Gaussian class"""
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
        that calculates the covariance kernel matrix between two matrices:
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

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points
        in a Gaussian process:
        :param X_s:is a numpy.ndarray of shape (s, 1)
        containing all of the points
         whose mean and standard deviation should be calculated
        :return: mu, sigma
        """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.reshape(-1), cov_s.diagonal()

    def update(self, X_new, Y_new):
        """
        Update Gaussian Process
        :param self:
        :param X_new: is a numpy.ndarray of shape (1,) that
        represents the new sample point
        :param Y_new: is a numpy.ndarray of shape (1,) that
        represents the new sample function value
        :return: Updates the public instance attributes X, Y, and K
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
