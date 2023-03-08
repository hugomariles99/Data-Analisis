import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import re
from pandas.plotting import scatter_matrix
import numpy as np

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 150)

df = pd.read_csv("Datasets/All_GPUs.csv")
#df = pd.DataFrame({'column_nd': [1, 25, 3, 7, 1, 12, 8, 11, 7], 'column_nc': [2.6, 3.5, 6.3, 8.9, 1.65, 11.5, 5.2, 10.9, 7.6], 'column_t':  ['a', 'b', 'a', 'c', 'a', 'c', 'b', 'c', 'a']}, columns=['column_nd', 'column_nc', 'column_t'])
#print (df)

""" 
******************** DATOS FALTANTES / OUTLIERS ******************** 

#print(); print("Datos faltantes: ", df.isnull().sum().sum())

#print(); print("Datos faltantes por columna:")
#print(df.isnull().sum().sort_values(ascending = False))

#df['column_nc'] = df['column_nc'].fillna('Examen')
#df = df.fillna("Unidad 1")

df = df.interpolate()

#df = df.dropna()
#print(); print (df)

#df = df.reset_index(drop = True)

#print(); print (df)
#print(); print("Datos faltantes: ", df.isnull().sum().sum())

#print(); print("Datos faltantes por columna:")
#print(df.isnull().sum().sort_values(ascending = False))
sns.boxplot(x=df['TMUs'])
"""

"""
******************** EXPLORACION ESTADISTICA ********************

plt.figure();
df['Shader'].plot(kind = 'bar');
df.plot(kind = 'box', y = ['DVI_Connection','Shader', 'Open_GL', 'VGA_Connection']);
df.plot(kind = 'kde', y = ['DVI_Connection','Shader', 'Open_GL', 'VGA_Connection']);
df.plot(kind = 'area', y = ['DVI_Connection','Shader', 'Open_GL', 'VGA_Connection']);
df.plot(legend=False)
"""

""" 
******************** ESCALAMIENTO ********************

df[["column_nd_e1"]] = MinMaxScaler().fit_transform(df[["column_nd"]])
df[["column_nd_e2"]] = MinMaxScaler(feature_range = (0, 100)).fit_transform(df[["column_nd"]])

print(df['column_nd'], df['column_nd_e1'], df['column_nd_e2'])

df.plot(kind = 'box', y = ['column_nd']);
df.plot(kind = 'box', y = ['column_nd_e1']);
df.plot(kind = 'box', y = ['column_nd_e2']);


df[["TMUs_scaled1"]] = MinMaxScaler().fit_transform(df[["TMUs"]])
df[["TMUs_scaled2"]] = MinMaxScaler(feature_range = (0, 50)).fit_transform(df[["TMUs"]])
print(df['TMUs'], df['TMUs_scaled1'], df['TMUs_scaled2'])

df.plot(kind = 'box', y = ['TMUs']);
df.plot(kind = 'box', y = ['TMUs_scaled1']);
df.plot(kind = 'box', y = ['TMUs_scaled2']);

"""

"""
******************** NORMALIZACION ********************

x = df[['TMUs']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)

df.plot(kind = 'bar', y = 'TMUs')
df_normalized.plot(kind = 'bar')

print(); print(df_normalized)

print(); print(df['TMUs'])
"""

"""
******************** ETIQUETADO ********************

#print(df)

le = preprocessing.LabelEncoder()
le.fit(df['Memory_Type'])
print(list(le.classes_))

newColumn = le.transform(df['Memory_Type'])
print(); print(newColumn)

print()
df.insert(18, "Memory_Type_LE", newColumn)

df.plot(kind = 'line', y = ['Memory_Type_LE'], xticks = 100, ylabel = "Etiqueta");
"""

"""
******************** REGEX ********************

ejemplo = 'Cuando cuentes cuentos, cuenta cuántos cuentos cuentas'
ejemplo2 = "abc"
match = re.search("cuent", ejemplo2)

if match:
    print("Match")
else:
    print("No match")

print("Nvidia = ", df['Manufacturer'].str.count(r'Nvidia').sum())
print("AMD = ",df['Manufacturer'].str.count(r'AMD').sum())
print("Intel = ", df['Manufacturer'].str.count(r'Intel').sum())
print("ATI = ", df['Manufacturer'].str.count(r'ATI').sum())
"""

"""
******************** MATRIZ DE CORRELACION ********************

#columnas = ['column_nd', 'column_nc']
plt.rcParams.update({'font.size': 8})

le = preprocessing.LabelEncoder()
le.fit(df['Memory_Type'])
print(list(le.classes_))
newColumn = le.transform(df['Memory_Type'])
df.insert(34, "Memory_Type_E", newColumn)

le.fit(df['Direct_X'])
print(list(le.classes_))
newColumn = le.transform(df['Direct_X'])
df.insert(35, "Direct_X_E", newColumn)

le.fit(df['Memory_Speed'])
print(list(le.classes_))
newColumn = le.transform(df['Memory_Speed'])
df.insert(35, "Memory_Speed_E", newColumn)

columnas = ['Direct_X_E', 
            'Memory_Type_E',
            'Memory_Speed_E']

scatter_matrix(df[columnas])
plt.show()
"""

"""
******************** CATEGORIAS ********************

category  = pd.Series(["a","b","c"], dtype="category")
print(category)
print(category.cat.add_categories(['d']))
print(category.cat.remove_categories("b"))

category  = pd.Categorical(df['Manufacturer'], categories = ["Nvidia"])
print(category)

df_categorias = df['Manufacturer']
df_categorias['Category'] = category

print(); print(df_categorias)
"""