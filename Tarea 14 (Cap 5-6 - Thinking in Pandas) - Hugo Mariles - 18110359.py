import warnings
warnings.filterwarnings("ignore")
   
import pandas as pd
import numpy as np
    
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 150)

df = pd.read_csv("Datasets/All_GPUs.csv")  

""" ****************************************** PIVOT TABLE ****************************************** """
#Calculating the average with pivot_table
""" WORKS
df2 = df.pivot_table(
    values=['Open_GL'],
    index=["HDMI_Connection","DisplayPort_Connection"],
    aggfunc=np.mean
    )
print(df2)
"""

#Calculating the average with groupby
""" ERROR 
df2 = df[['Open_GL']].groupby(["HDMI_Connection","DisplayPort_Connection"]).mean()
print(); print(df2)
"""

#Re-orienting a DataFrame using pivot
#Pivot throws a ValueError when there are multiple values for the same column-index combination
""" ERROR 
df3 = df.pivot(
    index="Open_GL",
    columns="TMUs",
    values="Open_GL"
    )
print(); print(df3)
ValueError: Index contains duplicate entries, cannot reshape
"""

""" ****************************************** STACK ****************************************** """

#Reshaping a DataFrame so that each row represents a count using stack
"""
df4 = df.stack().reset_index()
print(); print(df4)
#df4.drop(columns=["level_0"], inplace=True)
#print(); print(df4)
df4.set_index(["level_0"], inplace=True)
print(); print(df4)
"""

#Reshaping a DataFrame so that each column is a count using unstack
"""
df["Count_OpenGL"] = df.groupby("Open_GL").cumcount()

data = [df["HDMI_Connection"], df["DisplayPort_Connection"], df["Count_OpenGL"], df["Open_GL"]]
headers = ["HDMI_Connection", "DisplayPort_Connection", "Count_OpenGL", "Open_GL"]
df5 = pd.concat(data, axis=1, keys=headers)

print(); print(df5)

df5.set_index("Count_OpenGL", append=True, inplace=True)
df5 = df5.unstack()

print(); print(df5)
"""

""" ****************************************** MELT ****************************************** """
#Reshaping a DataFrame so that each row represents an inspection using melt
"""
data = [df["Name"], 
        df["Manufacturer"],  
        df["Memory"],  
        df["Memory_Speed"],
        df["Memory_Type"],
        ]
headers = ["Name", 
           "Manufacturer", 
           "Memory",
           "Memory_Speed",
           "Memory_Type"
           ]
df6 = pd.concat(data, axis = 1, keys = headers)
#print(df6)

df6 = df6.melt(
    id_vars=["Name","Memory"],
    value_vars=["Memory_Speed", "Memory_Type"],
    value_name="Memory Speed / Type").drop(columns="variable")
print(); print(df6)
"""

""" ****************************************** TRANSPOSE ****************************************** """
#Using a transpose to reshape a DataFrame
"""
data = [df["Name"], 
        df["Manufacturer"],  
        df["Memory"]
        ]
headers = ["Name", 
           "Manufacturer", 
           "Memory"
           ]
df7a = pd.concat(data, axis = 1, keys = headers)
print(df7a)

data = [df["Name"],
        df["Memory_Speed"],
        df["Memory_Type"]
        ]
headers = ["Name",
           "Memory_Speed",
           "Memory_Type"
           ]
df7b = pd.concat(data, axis = 1, keys = headers)
print(df7b)

#df7a = df7a.transpose(copy=False)
#print(df7a)

print(df7b.join(df7a.set_index('Name'), on='Name', how='left'))
"""

""" ****************************************** APPLY ****************************************** """
"""
data = [df["DVI_Connection"], 
        df["DisplayPort_Connection"],  
        df["HDMI_Connection"],
        df["VGA_Connection"]
        ]
headers = ["DVI_Connection", 
           "DisplayPort_Connection", 
           "HDMI_Connection",
           "VGA_Connection"
           ]
df8 = pd.concat(data, axis = 1, keys = headers)

#Example of using apply
#print(df8)
print(df8.apply(np.sum, axis=1))

print(); print(df8.sum(axis=1))
#df.sum() suma todos los datos numericos, no tira error si existen datos cualitativos
#print(); print(df.sum(axis=1))
"""

#Reemplazar datos con APPLY ------> por el max de las columnas indicadas
"""
print(df8)
def replace_missing(series):
    if np.isnan(series["DisplayPort_Connection"]):
        series["DisplayPort_Connection"] = max(series["DVI_Connection"], series["HDMI_Connection"])
        return series
df8 = df8.apply(replace_missing, axis=1)
print(); print(df8)
"""

#Reemplazar datos con WHERE ------> por el max de las columnas indicadas
"""
print(df8)
df8["DisplayPort_Connection"].where(
    ~df8["DisplayPort_Connection"].isna(),
    df8[["DVI_Connection", "HDMI_Connection"]].max(axis=1),
    inplace=True,
)
print(); print(df8)
"""

#Dropping rows whose "X" column does not contain the substring "Y" using APPLY
"""
data = [df["Name"], 
        df["Manufacturer"],
        df["Memory_Speed"],  
        df["Memory_Type"]
        ]
headers = ["Name", 
           "Manufacturer", 
           "Memory_Speed",
           "Memory_Type"
           ]
df9 = pd.concat(data, axis = 1, keys = headers)
print(df9)

def test_text_in_order(series):
    if ("GTX" in 
        series["Name"]
        ):
        return series
    return np.nan

df9 = df9.apply(
    test_text_in_order,
    axis=1,
    result_type="reduce",
    ).dropna()

df9 = df9.reset_index()
print(); print(df9)
"""

#Solving Listing ^^^ using a list comprehension
"""
data = pd.DataFrame({
    "fruit": ["orange", "lemon", "mango"],
    "order": [
        "I'd like an orange",
        "Mango please.",
        "May I have a mango?",
        ],
    })

mask = [fruit.lower() in order.lower()
for (fruit, order) in data[["fruit", "order"]].values]
data = data[mask]

print(data)
"""

#Implementation of scipy.stats.percentileofscore
""" No hace nada """
def percentileofscore(a, score):
    n = len(a)
    left = np.count_nonzero(a < score)
    right = np.count_nonzero(a <= score)
    pct = (right + left + (1 if right > left else 0)) * 50.0/n
    return pct

from scipy import stats
data = pd.DataFrame(np.arange(20).reshape(4,5))
print(data)

def apply_percentileofscore(series):
    return series.apply(
        lambda x:stats.percentileofscore(series,x)
        )

data.apply(apply_percentileofscore, axis=1)
print(); print(data)