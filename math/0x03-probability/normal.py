#!/usr/bin/env python3
""" Initialize Normal  """


class Normal:
    """create a class Normal that represents a normal distribution:"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            summatory = 0
            for elem in data:
                summatory += (elem - self.mean) ** 2
            self.stddev = (summatory / len(data)) ** 0.5

        else:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)

    def z_score(self, x):
        """ the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        return 1 / (self.stddev * (2 * 3.1415926536) ** 0.5) * \
            2.7182818285 ** (- (x - self.mean)**2 / (2 * self.stddev**2))

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        arg = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / 3.1415926536 ** 0.5) * \
              (arg - (arg ** 3) / 3 + (arg ** 5) / 10 -
               (arg ** 7) / 42 + (arg ** 9) / 216)
        return (1 / 2) * (1 + erf)
