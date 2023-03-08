# -*- coding: utf-8 -*-
import pandas as pd
import openpyxl

data_crypto = pd.read_csv("Datasets/Crypto.csv")

print(data_crypto)

data_crypto.to_excel(r"Datasets/Crypto_excel.xlsx")

data_excel = pd.read_excel("Datasets/Crypto_excel.xlsx")

print(data_excel)

data_excel.to_html(r"Datasets/Crypto_html.html")