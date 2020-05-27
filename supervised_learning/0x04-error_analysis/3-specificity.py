#!/usr/bin/env python3
""" Specificity"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix:
    :param confusion:
    :return: a numpy.ndarray of shape (classes,)
    containing the specificity of each class
    """
    TP = np.diagonal(confusion)
    TN = np.sum(TP) - TP
    FP = np.sum(confusion, axis=1) - TP
    speci = TN / (TN + FP)
    return speci
