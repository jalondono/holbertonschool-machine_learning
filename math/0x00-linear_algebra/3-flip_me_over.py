#!/usr/bin/env python3

"""  transpose of a 2D matrix, """


def matrix_transpose(matrix):
    """transpose of a 2D matrix"""
    flip_matrix = []
    for rows in matrix:
        for idx, column in enumerate(rows):
            if len(flip_matrix) != len(matrix[0]):
                flip_matrix.append([column])
            else:
                flip_matrix[idx].append(column)
    return flip_matrix
