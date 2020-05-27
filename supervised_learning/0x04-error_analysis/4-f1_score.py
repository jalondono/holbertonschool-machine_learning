#!/usr/bin/env python3
""" F1 score """
import numpy as np

precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix:
    :param confusion:
    :return:  a numpy.ndarray of shape (classes,)
     containing the F1 score of each class
    """
    precision_value = precision(confusion)
    sensitivity_value = sensitivity(confusion)
    a = (2 * precision_value * sensitivity_value)
    b = precision_value + sensitivity_value
    f1_value = a / b
    return f1_value
