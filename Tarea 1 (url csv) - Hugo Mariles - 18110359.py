# -*- coding: utf-8 -*-
import pandas as pd

""" DATA """
data_url_1 = pd.read_csv("https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv")
data_url_2 = pd.read_csv("https://raw.githubusercontent.com/hxchua/datadoubleconfirm/master/datasets/arrivals2018.csv")
data_crypto = pd.read_csv("Datasets/Crypto.csv")
""" DATA """

print("\n---------- DATA 1 ----------\n")
print(data_url_1)

print("\n---------- DATA 2 ----------\n")
print(data_url_2)
data_url_2.tail()

print("\n---------- DATA 3 ----------\n")
print(data_crypto)
print(data_crypto.describe())
