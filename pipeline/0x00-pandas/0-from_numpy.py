#!/usr/bin/env python3
""" From Numpy """
import pandas as pd


def from_numpy(array):
    """
    creates a pd.DataFrame from a np.ndarray
    :param array: np.ndarray
    :return: the newly created pd.DataFrame
    """
    col = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    col_shape = col[:array.shape[1]]
    return pd.DataFrame(array, columns=col_shape)
