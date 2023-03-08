import pandas as pd
import seaborn as sns

tips = sns.load_dataset("tips")
df = df = pd.read_csv("Datasets/All_GPUs.csv")  

#g = sns.lmplot(x="total_bill", y="tip", data=tips)

#g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

#g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, markers=['o', "x"])

#g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, palette="Set2")

#g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, palette=dict(Yes="g", No="r"))

#g = sns.lmplot(x="size", y="total_bill", hue="day", col="day", data=tips, height=6, aspect=.4, x_jitter=.1)

g = sns.lmplot(x = "Open_GL", y = "TMUs", col = "HDMI_Connection", hue = "HDMI_Connection", 
               data = df, col_wrap = 2, height = 6, aspect = 1, markers=['o', 'x', 's', '+'])

