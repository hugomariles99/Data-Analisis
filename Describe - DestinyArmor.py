import pandas as pd

df = pd.read_csv("Datasets/destinyArmor.csv")

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 26)
pd.set_option('display.width', 150)

print(df.describe(include='all'))