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
    # TP = np.diagonal(confusion)
    # TN = np.sum(TP) - TP
    # FP = np.sum(confusion, axis=0) - TP
    # speci = TN / (TN + FP)
    # return speci
    # Tiene sentido utilizar la diagonal la definicion
    # de abajo, no tiene sentido
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    SPECIFICITY = TN / (FP + TN)

    return SPECIFICITY
