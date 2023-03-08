import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
 
# I want 7 days of 24 hours with 60 minutes each 
periods = 7 * 24 * 60 
tidx = pd.date_range('2022-04-27', periods=periods, freq='T') 
#                     ^                                   ^ #                     |                                   | #                 Start Date        Frequency Code for Minute # This should get me 7 Days worth of minutes in a datetimeindex 
 
# Generate random data with numpy.  We'll seed the random 
# number generator so that others can see the same results. 
# Otherwise, you don't have to seed it. 

np.random.seed([990312]) 
 
# This will pick a number of normally distributed random numbers 
# where the number is specified by periods 

data = np.random.randn(periods) 
 
ts = pd.Series(data=data, index=tidx, name='Towy es el jefe') 
 
print(ts.describe());

ts.resample('15T').last()

ts.resample('15T').agg(['min', 'mean', 'max']).plot()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, freq in enumerate(['15T', '30T', '1H']):     
    ts.resample(freq).agg(['max', 'mean', 'min']).plot(ax=axes[i], title=freq)