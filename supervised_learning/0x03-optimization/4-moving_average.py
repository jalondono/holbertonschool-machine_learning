#!/usr/bin/env python3
"""Moving Average"""
import numpy as np


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set:
    :param data: is the list of data to calculate the moving average of
    :param beta: is the weight used for the moving average
    :return: a list containing the moving averages of data
    """
    vt = 0
    mov_average = []
    for idx in (range(len(data))):
        vt = beta * vt + ((1 - beta) * data[idx])
        bias = 1.0 - (beta ** (idx + 1))
        mov_average.append(vt / bias)
    return mov_average
