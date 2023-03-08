import pandas as pd

#df = pd.read_csv("Datasets/All_GPUs.csv")
df = pd.read_csv("Datasets/Examen Unidad 2 - Plantas/plants 3.data")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 150)

print(df)

print(df.describe())
print(df.describe(include='all'))

print("*****************************************************")

#print(df)

#print(df['Manufacturer'].unique())

#print(df.count().sum())

#print(); print("Datos faltantes: ", df.isnull().sum().sum())

#print(); print("Datos faltantes por columna:")
#print(df.isnull().sum().sort_values(ascending = False))

#for col in df.columns:
   # print(col)
   
#print(df['DisplayPort_Connection'])