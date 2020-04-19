#!/usr/bin/env python3
""" Slice Like A Ninja """


def np_slice(matrix, axes={}):
    """ slices a matrix along a specific axes:"""
    empty_slice = slice(None, None, None)
    aux_list = []
    for idx in range(0, matrix.ndim):
        if idx in axes:
            aux_list.append(slice(*axes[idx]))
        else:
            aux_list.append(empty_slice)
        tuple(aux_list)
    return matrix[tuple(aux_list)]
