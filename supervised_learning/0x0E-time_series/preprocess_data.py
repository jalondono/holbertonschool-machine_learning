#!/usr/bin/env python3
"""Preprocess data"""

import numpy as np
import tensorflow.keras as K
import pandas as kunfu
import datetime as dt
WindowGenerator = __import__('windows_generator').WindowGenerator


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
    cb_df = cb_df[cb_df['Timestamp'].dt.year >= 2018]
    cb_df.reset_index(inplace=True, drop=True)

    # bstamp_df = kunfu.read_csv(csv_bitstamp)
    # bstamp_df['Timestamp'] = kunfu.to_datetime(bstamp_df['Timestamp'], unit='s')
    # getting the closing date
    n = len(cb_df)

    # split the data to training, validation and testing
    train_df = cb_df[0:int(n*0.7)]
    val_df = cb_df[int(n*0.7):int(n*0.9)]
    test_df = cb_df[int(n*0.9):]

    # normalization of the data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df.iloc[:, 1:] = (train_df.iloc[:, 1:] - train_mean) / train_std
    val_df.iloc[:, 1:] = (val_df.iloc[:, 1:] - train_mean) / train_std
    test_df.iloc[:, 1:] = (test_df.iloc[:, 1:] - train_mean) / train_std

    w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                         train_df=train_df, val_df=val_df,
                         test_df=test_df, label_columns=['T (degC)'])
    print(w1)





if __name__ == '__main__':
    cb_path = '../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    bstamp_path = '../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    preprocess(cb_path, bstamp_path)
