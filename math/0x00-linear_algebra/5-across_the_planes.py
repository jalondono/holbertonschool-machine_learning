#!/usr/bin/env python3
""" adds two 2D matrices """

matrix_shape = __import__('2-size_me_please').matrix_shape
"""import module to find the shape """


def add_matrices2D(mat1, mat2):
    """adds two 2D matrices"""
    added_matrix = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    """if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    """
    for index in range(len(mat1)):
        added_matrix.append([x + y for x, y in zip(mat1[index], mat2[index])])
    return added_matrix
