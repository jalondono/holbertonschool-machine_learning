#!/usr/bin/env python3
""" Function to calculate the shape of a matrix"""


def matrix_shape(matrix):
    shape_matrix = []
    if matrix is None:
        return shape_matrix
    if len(matrix) > 0:
        shape_matrix.append(len(matrix))
        if type(matrix[0]) == int:
            return shape_matrix
        else:
            shape_matrix.append(len(matrix[0]))
            if type(matrix[0][0]) == int:
                return shape_matrix
            else:
                shape_matrix.append(len(matrix[0][0]))
    return shape_matrix
