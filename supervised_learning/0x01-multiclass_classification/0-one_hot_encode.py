#!/usr/bin/env python3
"""One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """
    hat converts a numeric label vector into a one-hot matrix:
    :param Y: containing numeric class labels
    :param classes: is the maximum number of classes found in Y
    :return: a one-hot encoding of Y with shape (classes, m),
     or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    b = np.zeros((Y.size, classes))
    b[Y, np.arange(Y.size)] = 1
    return b
