#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:
    :param confusion: confusion is a confusion numpy.ndarray of shape
    (classes, classes) where row indices represent the correct
     labels and column indices represent the predicted labels
    :return: the sensitivity of each class
    """

    return np.diagonal(confusion) / np.sum(confusion, axis=1)
