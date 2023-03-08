import pandas as pd
import seaborn as sns

df = sns.load_dataset("tips")

#print(df)
print(); print(df.describe())

""" ********** TOTAL_BILL ********** """
tb_mean = df.total_bill.mean()
tb_min = df.total_bill.min()
tb_25 = df.total_bill.quantile(0.25)
tb_50 = df.total_bill.quantile(0.5)
tb_75 = df.total_bill.quantile(0.75)
tb_max = df.total_bill.max()

""" ********** TIP ********** """
tip_mean = df.tip.mean()
tip_min = df.tip.min()
tip_25 = df.tip.quantile(0.25)
tip_50 = df.tip.quantile(0.5)
tip_75 = df.tip.quantile(0.75)
tip_max = df.tip.max()

""" ********** SIZE ********** """
size_mean = df['size'].mean()
size_min = df['size'].min()
size_25 = df['size'].quantile(0.25)
size_50 = df['size'].quantile(0.5)
size_75 = df['size'].quantile(0.75)
size_max = df['size'].max()

#.sub(search_value) subtracts search_value from the df[col_to_search] to make the nearest value almost-zero,
#.abs() makes the almost-zero the minimum of the column,
#.idxmin() yields the df.index of the minimum value, or the closest match to search_value.

print(); print(format("MEAN","*^100"));
print(); print(format("total_bill","-^50"));
tb_index_mean = df['total_bill'].sub(tb_mean).abs().idxmin()
print(df.loc[[tb_index_mean]])

print(); print(format("tip","-^50"));
tip_index_mean = df['tip'].sub(tip_mean).abs().idxmin()
print(df.loc[[tip_index_mean]])

print(); print(format("size","-^50"));
size_index_mean = df['size'].sub(size_mean).abs().idxmin()
print(df.loc[[size_index_mean]])

""" -------------------------------------------------------- """

print(); print(format("MIN","*^100"));
print(); print(format("total_bill","-^50"));
tb_index_min = df['total_bill'].sub(tb_min).abs().idxmin()
print(df.loc[[tb_index_min]])

print(); print(format("tip","-^50"));
tip_index_min = df['tip'].sub(tip_min).abs().idxmin()
print(df.loc[[tip_index_min]])

print(); print(format("size","-^50"));
size_index_min = df['size'].sub(size_min).abs().idxmin()
print(df.loc[[size_index_min]])

""" -------------------------------------------------------- """

print(); print(format("25%","*^100"));
print(); print(format("total_bill","-^50"));
tb_index_25 = df['total_bill'].sub(tb_25).abs().idxmin()
print(df.loc[[tb_index_25]])

print(); print(format("tip","-^50"));
tip_index_25 = df['tip'].sub(tip_25).abs().idxmin()
print(df.loc[[tip_index_25]])

print(); print(format("size","-^50"));
size_index_25 = df['size'].sub(size_25).abs().idxmin()
print(df.loc[[size_index_25]])

""" -------------------------------------------------------- """

print(); print(format("50%","*^100"));
print(); print(format("total_bill","-^50"));
tb_index_50 = df['total_bill'].sub(tb_50).abs().idxmin()
print(df.loc[[tb_index_50]])

print(); print(format("tip","-^50"));
tip_index_50 = df['tip'].sub(tip_50).abs().idxmin()
print(df.loc[[tip_index_50]])

print(); print(format("size","-^50"));
size_index_50 = df['size'].sub(size_50).abs().idxmin()
print(df.loc[[size_index_50]])

""" -------------------------------------------------------- """

print(); print(format("75%","*^100"));
print(); print(format("total_bill","-^50"));
tb_index_75 = df['total_bill'].sub(tb_75).abs().idxmin()
print(df.loc[[tb_index_75]])

print(); print(format("tip","-^50"));
tip_index_75 = df['tip'].sub(tip_75).abs().idxmin()
print(df.loc[[tip_index_75]])

print(); print(format("size","-^50"));
size_index_75 = df['size'].sub(size_75).abs().idxmin()
print(df.loc[[size_index_75]])

""" -------------------------------------------------------- """

print(); print(format("MAX","*^100"));
print(); print(format("total_bill","-^50"));
tb_index_max = df['total_bill'].sub(tb_max).abs().idxmin()
print(df.loc[[tb_index_max]])

print(); print(format("tip","-^50"));
tip_index_max = df['tip'].sub(tip_max).abs().idxmin()
print(df.loc[[tip_index_max]])

print(); print(format("size","-^50"));
size_index_max = df['size'].sub(size_max).abs().idxmin()
print(df.loc[[size_index_max]])

""" -------------------------------------------------------- """

#print(); print(df.loc[[154]])
#array = ['tb_mean', 'tb_min']
#df.loc[df['total_bill'] == tb_min]