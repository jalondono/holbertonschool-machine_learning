#!/usr/bin/env python3
"""Preprocess data"""

import numpy as np
import tensorflow.keras as K
import pandas as kunfu
import datetime as dt
import tensorflow as tf

WindowGenerator = __import__('windows_generator').WindowGenerator
Baseline = __import__('windows_generator').Baseline


def preprocess(csv_coin_base, csv_bitstamp):
    """
    clean tha data from csv
    :param csv_coin_base: dataset
    :param csv_bitstamp: dataset
    :return: One dataframe with the cleaned data
    """
    cb_df = kunfu.read_csv(csv_coin_base,
                           nrows=2000000).dropna()
    cb_df['Timestamp'] = kunfu.to_datetime(cb_df['Timestamp'], unit='s')
    cb_df = cb_df[cb_df['Timestamp'].dt.year >= 2017]
    cb_df.reset_index(inplace=True, drop=True)
    cb_df = cb_df.drop(['Timestamp'], axis=1)
    cb_df = cb_df[0::60]

    n = len(cb_df)

    # split the data to training, validation and testing
    train_df = cb_df[0:int(n * 0.7)]
    val_df = cb_df[int(n * 0.7):int(n * 0.9)]
    test_df = cb_df[int(n * 0.9):]

    # normalization of the data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
