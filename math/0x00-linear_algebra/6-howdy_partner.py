#!/usr/bin/env python3
""" Howdy Partner """


def cat_arrays(arr1, arr2):
    """ concatenates two arrays"""
    aux = []
    aux = arr1.copy()
    aux.extend(arr2)
    return aux
