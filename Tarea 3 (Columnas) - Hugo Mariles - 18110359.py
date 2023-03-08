# -*- coding: utf-8 -*-
import pandas as pd
import openpyxl

data_crypto = pd.read_csv("Datasets/Crypto.csv")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

print(data_crypto.shape)

print(data_crypto)

