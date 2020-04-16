#!/usr/bin/env python3
""" Ridin’ Bareback   """


def get_column(matrix, i):
    """Return a list of column of a 2d Array"""
    return [row[i] for row in matrix]


def multiplyList(arr1, arr2):
    """Return the sum of the product of two list"""
    total = 0
    for i in range(0, len(arr1)):
        total += arr1[i] * arr2[i]
    return total


def mat_mul(mat1, mat2):
    """ that performs matrix multiplication: """
    if len(mat1[0]) != len(mat2):
        return None
    multiply_list = []
    current_column = []
    for row_mat1 in mat1:
        aux_list = []
        for idx in range(len(mat2[0])):
            current_column = get_column(mat2, idx)
            aux_list.append(multiplyList(row_mat1, current_column))
        multiply_list.append(aux_list)
    return multiply_list
