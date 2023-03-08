import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing

import pandas as pd

df = pd.read_csv("Datasets/destinyArmor.csv")

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

#df.drop(df.iloc[:, 8:53], inplace = True, axis = 1)
print(df)

print("-------------------------------------------------------------------\n")
le = preprocessing.LabelEncoder()

categ = ['Type','Tier']

# Encode Categorical Columns
df[categ] = df[categ].apply(le.fit_transform)

le.fit(df['Type'])
print(list(le.classes_))

newColumn = le.transform(df['Type'])
print(); print(newColumn)

print()
df.insert(6, "Label_Type", newColumn)

print(df)