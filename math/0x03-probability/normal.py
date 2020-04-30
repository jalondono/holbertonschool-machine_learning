#!/usr/bin/env python3
""" Initialize Normal  """


class Normal:
    """create a class Normal that represents a normal distribution:"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            summatory = 0
            for elem in data:
                summatory += (elem - self.mean) ** 2
            self.stddev = (summatory / len(data)) ** 0.5

        else:
            self.mean = float(mean)
            if stddev < 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
