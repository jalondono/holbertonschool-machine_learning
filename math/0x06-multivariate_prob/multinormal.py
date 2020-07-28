#!/usr/bin/env python3
"""Initialize """
import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov
# corr = __import__('1-correlation').correlation


class MultiNormal:
    def __init__(self, data):
        """
        represents a Multivariate Normal distribution
        :param data: is a numpy.ndarray of shape (d, n)
        containing the data set:
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a 2D numpy.ndarray')
        if len(data.shape) != 2:
            err = 'data must contain multiple data points'
            raise ValueError(err)
        d, n = data.shape

        if n < 2:
            err = 'data must contain multiple data points'
            raise ValueError(err)

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        deviaton = data - self.mean
        self.cov = np.matmul(deviaton, deviaton.T) / (n - 1)
