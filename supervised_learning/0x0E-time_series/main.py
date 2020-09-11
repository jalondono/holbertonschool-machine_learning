#!/usr/bin/env python3
"""Main"""

import numpy as np
import tensorflow.keras as K
import pandas as kunfu
import datetime as dt
import tensorflow as tf
WindowGenerator = __import__('windows_generator').WindowGenerator
preprocess = __import__('preprocess_data').preprocess
Baseline = __import__('windows_generator').Baseline
forecast = __import__('forecast_btc').forecast

if __name__ == '__main__':
    val_performance = {}
    performance = {}

    cb_path = '../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    bstamp_path = '../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    train_df, val_df, test_df = preprocess(cb_path, bstamp_path)

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=train_df, val_df=val_df,
        test_df=test_df, label_columns=['Close'])

    lstm_model = forecast(wide_window)
    print(lstm_model)

    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
    wide_window.plot(lstm_model)
