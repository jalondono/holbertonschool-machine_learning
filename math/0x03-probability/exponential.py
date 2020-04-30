#!/usr/bin/env python3
"""Initialize Exponential """


class Exponential:
    """class Exponential that represents an exponential distribution:"""

    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(1 / (sum(data) / len(data)))
        else:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        e = 2.7182818285
        if x < 0:
            return 0
        pdf_value = (self.lambtha * (e**(-self.lambtha*x)))
        return pdf_value

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - (e**(-self.lambtha*x))
