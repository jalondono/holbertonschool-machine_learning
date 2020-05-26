#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:
    :param confusion:
    :return:
    """
    idx_max_value = np.argmax(confusion, axis=0)
    sumatory_classes = np.sum(confusion, axis=1)
    senstivity_classes = []
    b = confusion[0][0]

    for idx in range(confusion.shape[0]):
        senstivity_classes.append(
            confusion[idx][idx_max_value[idx]] / sumatory_classes[idx])
    return senstivity_classes
