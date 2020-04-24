#!/usr/bin/env python3
""". Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    derivate = []
    """ that calculates the derivative of a polynomial:"""
    if poly is None:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        derivate.append(i * poly[i])
    return derivate
