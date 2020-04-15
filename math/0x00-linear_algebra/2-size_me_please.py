#!/usr/bin/env python3

""" Size Me Please """


def matrix_shape(matrix):
    """ Function to calculate the shape of a matrix"""
    shape_matrix = []
    type_element = matrix[:]

    while type(type_element) == list:
        shape_matrix.append(len(type_element))
        type_element = type_element[0]
    return shape_matrix
