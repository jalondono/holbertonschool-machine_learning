#!/usr/bin/env python3
"""Adjugate"""


def check_shape(matrix):
    """
    Check if is a correct matrix
    :param matrix: matrix list
    :return:
    """
    if len(matrix):
        if not len(matrix[0]):
            return 3
    if not isinstance(matrix, list) or len(matrix) == 0:
        return 2
    for row in matrix:
        if len(row) != len(matrix):
            return 1
        if not isinstance(row, list):
            return 2
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
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a * d) - (b * c)
    det = 0
    for idx, data in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        n_m = [[val for i, val in enumerate(row) if i != idx] for row in rows]
        det += data * (-1) ** idx * determinant(n_m)
    return det


def minor(matrix):
    """
    that calculates the minor matrix of a matrix:
    :param matrix: list of lists whose minor matrix should be calculated
    :return:
    """

    err = 'matrix must be a list of lists'
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError(err)

    for element in matrix:
        if not isinstance(element, list):
            raise TypeError(err)

    err = 'matrix must be a non-empty square matrix'
    my_len = len(matrix)
    if my_len == 1 and len(matrix[0]) == 0:
        raise ValueError(err)

    for element in matrix:
        if len(element) != my_len:
            raise ValueError(err)

    if my_len == 1:
        return [[1]]

    final_mdet = []
    det = 0
    for index, each_row in enumerate(matrix):
        out_det = []
        rows = matrix[:index] + matrix[index + 1:]
        for idx, data in enumerate(each_row):
            n_m = [[v for i, v in enumerate(row) if i != idx] for row in rows]
            det = determinant(n_m)
            out_det.append(det)
        final_mdet.append(out_det)
    return final_mdet


def cofactor(matrix):
    """
    calculates the cofactor matrix of a matrix:
    :param matrix: list of lists whose cofactor matrix should be calculated
    :return: the cofactor matrix of matrix
    """
    my_len = len(matrix)

    my_minor = minor(matrix)

    my_cofactor = []
    for i in range(my_len):
        my_cofactor.append([])
        for j in range(my_len):
            sign = (-1) ** (i + j)
            value = sign * my_minor[i][j]
            my_cofactor[i].append(value)

    return my_cofactor


def adjugate(matrix):
    """
    is a list of lists whose adjugate matrix should be calculated
    :param matrix:
    :return:
    """
    my_cofactor = cofactor(matrix)
    my_adjugate = list(map(list, zip(*my_cofactor)))
    return my_adjugate
