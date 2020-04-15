#!/usr/bin/env python3
""" adds two 2D matrices """

matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """adds two 2D matrices"""
    added_matrix = []
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    for index in range(len(mat1)):
        added_matrix.append([x + y for x,y in zip(mat1[index], mat2[index])])
    return added_matrix
