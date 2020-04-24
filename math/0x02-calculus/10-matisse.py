#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """ that calculates the derivative of a polynomial:"""

    derivate = []
    if type(poly) != list:
        return None

    if poly is None or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    for element in poly:
        if not isinstance(element, (int, float)):
            return None

    for i in range(1, len(poly)):
        derivate.append(i * poly[i])
    return derivate
