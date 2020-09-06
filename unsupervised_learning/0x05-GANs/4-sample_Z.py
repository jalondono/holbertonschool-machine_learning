#!/usr/bin/env python3
"""Sampling"""
import numpy as np


def sample_Z(m, n):
    """
    creates input for the generator:
    :param m: is the number of samples that should be generated
    :param n: is the number of dimensions of each sample
    :return: Z, a numpy.ndarray of shape (m, n) containing the uniform samples
    """
    return np.random.uniform(-1., 1., size=[m, n])
