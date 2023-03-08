import pandas as pd

df = pd.read_csv("Datasets/destinyArmor.csv")
    
print(df)

def nan_columns(dataSetX):
    df = dataSetX
    for column_name in df.columns:
        row_count = len(df.index)
        column_count = df[column_name].isnull().sum()
        if column_count == row_count:
            #print (column_name, "-", column_count)
            del df[column_name]
        
nan_columns(df)

pd.set_option('display.max_rows', 20)

df1 = df[
    (df['Mobility (Base)'] > 10) 
    &
    (df['Recovery (Base)'] > 10)
    &
    (df['Total'] > 80)
    &
    (df['Equippable'] == "Warlock")
    ]

df2 = df[
    (df['Total'] > 100)
    &
    (df['Tier'] == "Exotic")
    ]

nan_columns(df1)
nan_columns(df2)

print(df1)
print(df2)