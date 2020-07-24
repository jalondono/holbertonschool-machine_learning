#!/usr/bin/env python3
"""Determinant"""


def check_shape(matrix):
    # 0 for True , 1 for Bad not matrix square,2 for not list of list
    if len(matrix):
        if not len(matrix[0]):
            return 3
    if not isinstance(matrix, list):
        return 2
    for row in matrix:
        if len(row) != len(matrix):
            return 1
    return 0


def determinant(matrix):
    """
     that calculates the determinant of a matrix:
    :param matrix:
    :return:
    """
    if check_shape(matrix) == 3:
        return 1
    if check_shape(matrix) == 2:
        raise TypeError("matrix must be a list of lists")
    if check_shape(matrix) == 1:
        raise TypeError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a*d) - (b*c)
    det = 0
    for idx, data in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        n_m = [[val for i, val in enumerate(row) if i != idx] for row in rows]
        det += data * (-1) ** idx * determinant(n_m)
    return det
