#!/usr/bin/env python3
""" adds two arrays element-wise:"""


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """
    added_array = []
    if len(arr1) != len(arr2):
        return None
    added_array = [x + y for x, y in zip(arr1, arr2)]
    return added_array
