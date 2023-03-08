import pandas as pd  
  
df = pd.read_csv("Datasets/All_GPUs.csv")  

#print(df)
print(df.describe())
#print(df.describe(include='all'))

#openGL = df['Open_GL']

#print(openGL)

pd.options.display.max_columns = 10
pd.options.display.max_rows = 10
pd.options.display.width = 150

#print(df.groupby('TMUs').mean())
print()

#print(df.groupby('Open_GL', sort = False).sum())
print()

#print(df.groupby(['TMUs', 'Open_GL']).sum())
print()

#print(df.groupby(['TMUs', 'Open_GL', 'Shader']).sum())
print()

#group = df.groupby('TMUs')
#for key, item in group:
    #print(group.get_group(key), "\n\n")

#print(df.groupby('TMUs').get_group(4.0))
print()

#df.groupby('Open_GL')['Open_GL'].plot(legend=False)

#df.groupby('TMUs').get_group(2.0)['Open_GL'].plot(legend=True)

df['Columna_GroupBy'] = df['TMUs'].groupby(df['Open_GL']).transform('sum')
print(df.describe())