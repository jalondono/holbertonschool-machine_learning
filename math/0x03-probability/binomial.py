#!/usr/bin/env python3
""" Initialize Binomial  """


class Binomial:
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            summatory = 0
            for elem in data:
                summatory += (elem - mean) ** 2
            variance = summatory / len(data)
            self.p = -1 * ((variance / mean) - 1)
            n = mean / self.p
            self.n = round(n)
            self.p *= n / self.n
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        k = int(k)
        if k < 0:
            return 0
        return (factorial(self.n) / factorial(k) / factorial(self.n - k)
                * self.p ** k * (1 - self.p) ** (self.n - k))


def factorial(m):
    """
    Calculates factorial of a number
    """
    if m == 1 or m == 0:
        return 1
    else:
        return m * factorial(m - 1)
