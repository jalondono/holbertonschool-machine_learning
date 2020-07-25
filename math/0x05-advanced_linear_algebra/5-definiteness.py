#!/usr/bin/env python3
"""Definiteness"""
import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix:
    :param matrix: is a numpy.ndarray of shape
     (n, n) whose definiteness should be calculated
    :return:the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite,
    or Indefinite if the matrix is positive definite,
     positive semi-definite, negative semi-definite,
     negative definite of indefinite, respectively
    """
    err = 'matrix must be a numpy.ndarray'
    if not isinstance(matrix, np.ndarray):
        raise TypeError(err)

    # square test
    my_len = matrix.shape[0]
    if len(matrix.shape) != 2 or my_len != matrix.shape[1]:
        return None

    # symmetry test
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    # list of sub matrices for upper left determinants calculation
    sub_matrices = [matrix[:i, :i] for i in range(1, my_len + 1)]

    # n upper left determinants
    det = np.array([np.linalg.det(sub_m) for sub_m in sub_matrices])
    odd = np.array([det[i - 1]
                    for i in range(1, det.shape[0] + 1) if i % 2 != 0])
    even = np.array([det[i - 1]
                     for i in range(1, det.shape[0] + 1) if i % 2 == 0])

    if all(det > 0):
        return 'Positive definite'
    elif all(det[:-1] > 0) and det[-1] == 0:
        return 'Positive semi-definite'
    elif all(det[:-1] < 0) and det[-1] == 0:
        return 'Negative semi-definite'
    elif all(odd < 0) and all(even > 0):
        return 'Negative definite'
    else:
        return 'Indefinite'