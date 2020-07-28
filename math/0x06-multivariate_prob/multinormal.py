#!/usr/bin/env python3
"""Initialize """
import numpy as np


class MultiNormal:
    """
    Multinormal class
    """

    def __init__(self, data):
        """
        represents a Multivariate Normal distribution
        :param data: is a numpy.ndarray of shape (d, n)
        containing the data set:
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            err = 'data must contain multiple data points'
            raise ValueError(err)

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        dev = data - self.mean
        self.cov = np.matmul(dev, dev.T) / (n - 1)
