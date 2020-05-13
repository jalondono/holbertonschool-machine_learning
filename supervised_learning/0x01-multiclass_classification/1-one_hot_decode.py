#!/usr/bin/env python3
"""One-Hot Decode"""

import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels:
    :param one_hot: numpy.ndarray with shape (classes, m)
    :return:  a numpy.ndarray with shape (m, ) containing
     the numeric labels for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    classes, m = one_hot.shape
    if np.sum(one_hot) != m:
        return None

    Y = np.zeros(m)
    tmp = np.arange(m)

    axis = np.argmax(one_hot, axis=0)
    Y[tmp] = axis
    return Y.astype("int64")
