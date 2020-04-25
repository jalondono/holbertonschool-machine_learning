#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """that calculates the integral of a polynomial: """
    integrate_list = []

    if type(poly) != list:
        return None

    if poly is None or len(poly) == 0:
        return None

    if poly == [0]:
        return [C]

    if not isinstance(C, int):
        return None

    for element in poly:
        if not isinstance(element, (int, float)):
            return None
    integrate_list.append(C)
    integrate_value = 0
    for idx in range(0, len(poly)):
        integrate_value = poly[idx] / (idx + 1)
        if integrate_value % 1 != 0:
            integrate_list.append(integrate_value)
        else:
            integrate_list.append(int(integrate_value))
    return integrate_list
