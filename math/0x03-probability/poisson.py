#!/usr/bin/env python3
""" Initialize Poisson """


class Poisson:
    """ Class Poisson """

    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        e = 2.7182818285
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for idx in range(1, k + 1):
            factorial = factorial * idx
        pmf_val = (e**(-self.lambtha) * (self.lambtha**k)) / factorial
        return pmf_val

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        sumatory = 0
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        for idx in range(k + 1):
            sumatory += self.pmf(idx)
        return sumatory
