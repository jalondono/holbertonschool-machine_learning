
#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file


def open_d(array):
    """

    :param array:
    :return:
    """
    return array[0]


def close_d(array):
    """

    :param array:
    :return:
    """
    return array[-1]


df = from_file("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ',')
df = df.loc[df["Timestamp"] >= 1483228800]
df = df.drop(["Weighted_Price"], axis=1)
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df = df.set_index("Date")

df["Close"].fillna(method="ffill", inplace=True)
df["Volume_(BTC)"].fillna(value=0, inplace=True)
df["Volume_(Currency)"].fillna(value=0, inplace=True)
df["High"].fillna(value=df.Close.shift(1, axis=0), inplace=True)
df["Low"].fillna(value=df.Close.shift(1, axis=0), inplace=True)
df["Open"].fillna(value=df.Close.shift(1, axis=0), inplace=True)

df_d = pd.DataFrame()
df_d["High"] = df.High.resample('D').max()
df_d["Low"] = df.Low.resample('D').min()
df_d["Volume_(BTC)"] = df["Volume_(BTC)"].resample('D').sum()
df_d["Volume_(Currency)"] = df["Volume_(Currency)"].resample('D').sum()
df_d["Open"] = df.Open.resample('D').apply(open_d)
df_d["close"] = df.Close.resample('D').apply(close_d)

df_d.plot()
plt.show()
