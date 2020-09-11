#!/usr/bin/env python3
"""Preprocess data"""

import numpy as np
import tensorflow.keras as K
import pandas as kunfu
import datetime as dt
import tensorflow as tf


def compile_and_fit(model, window, patience=2, Max_Epoch=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=Max_Epoch,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def forecast(wide_window):
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)])
    history = compile_and_fit(lstm_model, wide_window, Max_Epoch=25)
    return lstm_model
