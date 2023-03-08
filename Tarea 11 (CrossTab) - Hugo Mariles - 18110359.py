import warnings
warnings.filterwarnings("ignore")
   
import pandas as pd
    
pd.options.display.max_columns = 10
pd.options.display.max_rows = 20
pd.options.display.width = 150

df = pd.read_csv("Datasets/All_GPUs.csv")  
    
#print(); print(df)
print(df.describe())
#print(df.describe(include='all'))

df1 = pd.crosstab(df.TMUs, df.Open_GL, rownames = ['ROW'], colnames = ['COL'], margins=True)
#print(); print(df1)

df2 = pd.crosstab([df.TMUs, df.Open_GL], df.Shader,  margins=True)
#print(); print(df2)

#Comparando un par de columnas en ambos ejes, despues comparando esas comparaciones
df3 = pd.crosstab([df.TMUs, df.Open_GL], [df.Shader, df.HDMI_Connection], margins=True)
#print(); print(df3)
 
df4 = pd.crosstab(df.DisplayPort_Connection, df.HDMI_Connection, rownames = ['DP'], colnames = ['HDMI'], margins=True, dropna = True)
print(); print(df4)

#rot; cambia la rotacion del eje x
df4.plot.bar(rot=0, legend = True)

df1.to_csv('Datasets/Crosstab_Example.csv')
