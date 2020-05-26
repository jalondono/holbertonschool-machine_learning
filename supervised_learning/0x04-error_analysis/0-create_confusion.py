#!/usr/bin/env python3
"""Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix:
    :param labels: is a one-hot numpy.ndarray of shape (m, classes)
     containing the correct labels for each data point
    :param logits: is a one-hot numpy.ndarray of shape
    (m, classes) containing the predicted labels
    :return: a confusion numpy.ndarray of shape (classes, classes)
     with row indices representing the correct labels and column indices representing the predicted labels
    """

    confusion = np.matmul(labels.T, logits)
    return confusion
