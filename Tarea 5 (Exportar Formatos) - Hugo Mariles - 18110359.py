# -*- coding: utf-8 -*-
import pandas as pd

data_crypto = pd.read_csv("Datasets/Crypto.csv")

""" EXCEL """
data_crypto.to_excel(r"Tarea 5/Crypto.xlsx")
data_excel = pd.read_excel("Tarea 5/Crypto.xlsx")
print("\n---------- EXCEL ----------")
print(data_excel)

""" HDF """
data_crypto.to_hdf(r"Tarea 5/Crypto.h5", key = "crypto", mode = "w")
data_hdf = pd.read_hdf("Tarea 5/Crypto.h5", "crypto")
print("\n---------- HDF ----------")
print(data_hdf)

""" JSON """
data_crypto.to_json(r"Tarea 5/Crypto.json", orient="columns")
data_json = pd.read_json("Tarea 5/Crypto.json")
print("\n---------- JSON ----------")
print(data_json)

""" HTML """
data_crypto.to_html(r"Tarea 5/Crypto.html")
data_html = pd.read_html("Tarea 5/Crypto.html")
print("\n---------- HTML ----------")
print(data_html)

""" STATA """
data_crypto.to_stata(r"Tarea 5/Crypto.dta")
data_stata = pd.read_stata("Tarea 5/Crypto.dta")
print("\n---------- STATA ----------")
print(data_stata)

""" CLIPBOARD """
data_crypto.to_clipboard(sep=',')
data_clipboard = pd.read_clipboard(sep='\\s+')
print("\n---------- CLIPBOARD ----------")
print(data_clipboard)

""" PICKLE """
data_crypto.to_pickle(r"Tarea 5/Crypto.pkl")
data_pickle = pd.read_pickle("Tarea 5/Crypto.pkl")
print("\n---------- PICKLE ----------")
print(data_pickle)