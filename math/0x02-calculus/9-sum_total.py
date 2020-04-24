#!/usr/bin/env python3
"""Our life is the sum total of all the decisions we make every day"""


def summation_i_squared(n):
    """that calculates [sum_{i=1}^{n} i^2] :"""
    if n is None:
        return None
    result = (n * ((n + 1) * (2 * n + 1))) / 6
    return int(result)
