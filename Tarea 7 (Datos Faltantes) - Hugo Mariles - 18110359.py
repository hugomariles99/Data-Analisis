import warnings
warnings.filterwarnings("ignore")

import pandas as pd

df = pd.read_csv("Datasets/destinyArmor.csv")

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

print(df)

def null_check_full():
    print(df.isnull().values.any())

def null_check_full_fast():
    if df.isnull().values.sum() > 0:
        print("True")
    else:
        print("False")
        
def null_check_column(name):
    print(df[name].isnull().values.any())
            
def null_count_full():
    print(df.isnull().sum().sum())

def null_count_column(name):
    print(df[name].isnull().sum())

print()
null_check_full()
null_check_full_fast()
null_check_column('Name')
null_count_full()
null_count_column('Perks 15')

def fill_nan_zero(name):
    df[name] = df[name].fillna(0)
    
def fill_nan_value(name, value):
    df[name] = df[name].fillna(value)

#fill_nan_zero('Perks 15')
fill_nan_value('Perks 15', 'HM')

def drop_nan():
    #df.dropna(axis = 'columns', inplace = True)
    #df.dropna(how = 'all')
    #df.dropna(inplace = True)
    df.dropna(thresh = 45, inplace = True)
    
drop_nan()
print(df)

''' Reset Index'''
#df = df.reset_index(drop = True)