import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("Datasets/destinyArmor.csv")
#print(); print(df)

print("\n", format("TABLA NORMAL","*^100"), "\n", df["Total"])

x = df[['Total']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
print("\n", format("NORMALIZACION POR MINIMO - MAXIMO","*^100"), "\n", df_normalized);

df_max_scaled = df.copy()
df_max_scaled["Total"] = df_max_scaled["Total"] / df_max_scaled["Total"].abs().max()     
print("\n", format("NORMALIZACION POR MAXIMO","*^100"), "\n", df_max_scaled["Total"]);

df_z_scaled = df.copy()
df_z_scaled["Total"] = (df_z_scaled["Total"] - 
                        df_z_scaled["Total"].mean()) / df_z_scaled["Total"].std()   
print("\n", format("NORMALIZACION POR Z","*^100"), "\n", df_z_scaled["Total"]);