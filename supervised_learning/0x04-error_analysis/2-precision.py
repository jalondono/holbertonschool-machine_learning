#!/usr/bin/env python3
""" Precision"""
import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix:
    :param confusion:is a confusion numpy.ndarray of shape
    (classes, classes) where row indices represent the correct
     labels and column indices represent the predicted labels
    :return:
    """
    a = np.sum(confusion, axis=0)
    b = np.sum(confusion, axis=1)
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
