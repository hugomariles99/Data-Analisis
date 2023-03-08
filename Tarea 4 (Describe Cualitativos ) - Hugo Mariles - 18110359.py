# -*- coding: utf-8 -*-
import pandas as pd
import openpyxl

data_crypto = pd.read_csv("Datasets/Crypto.csv")

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

print(data_crypto.shape)

print(data_crypto)

print('\n' +
      '-----------------------------------------------------------------' +
      '-----------------------------------------------------------------\n')

print(data_crypto.describe(include='all'))
# Describe todas las columnas sin importar el tipo de datos