#!/usr/bin/env python3
""" From File """
import pandas as pd


def from_file(filename, delimiter):
    """
    loads data from a file as a pd.DataFrame:
    :param filename:
    :param delimiter:
    :return:
    """
    return pd.read_csv(filename, sep=delimiter)
