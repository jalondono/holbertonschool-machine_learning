#!/usr/bin/env python3
""" Gettin’ Cozy  """


def copy2dArray(array):
    """  Copy a 2dArray:"""
    copy_array = []
    for row in array:
        aux_list = []
        for column in row:
            aux_list.append(column)
        copy_array.append(aux_list)
    return copy_array


def cat_matrices2D(mat1, mat2, axis=0):
    """  that concatenates two matrices along a specific axis:"""
    aux_mat1 = copy2dArray(mat1)
    aux_mat2 = copy2dArray(mat2)
    if axis == 0:
        if len(aux_mat1[0]) == len(aux_mat2[0]):
            aux_mat1.extend(aux_mat2)
            return aux_mat1
        else:
            return None
    else:
        if len(aux_mat1) == len(aux_mat2):
            for idx, elem in enumerate(aux_mat1):
                elem.extend(aux_mat2[idx])
        else:
            return None
    return aux_mat1
