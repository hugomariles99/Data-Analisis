import sys
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from kneed import KneeLocator

df = pd.read_csv("Datasets/Examen Unidad 2 - Plantas/plants 4.data")
df2 = pd.read_csv("Datasets/Examen Unidad 2 - Plantas/states.txt")

#print(df.describe())
#print(df.describe(include='all'))

df['id'] = df.index

#print(df)
#print(df2)

# *******************************************************************************
"""
np.set_printoptions(threshold=sys.maxsize)

states = [0] * 69
s = "e"
for x in range(69):
    states[x] = (s + str(x+1))

    
a = df['e1']
estado_iniciales = 'ak'
arr_index = np.where(a == estado_iniciales)

location_string = str(arr_index)
index_a1 = location_string.index('[')
index_a2 = location_string.index(']')
location_string_2 = location_string[(index_a1 + 1):index_a2]
location_numbers = location_string_2.replace(", ", ".")
location_numbers += "."

v1 = location_numbers.count('.')
ls = location_numbers

numbers = [0] * v1

for x in range(v1):
    in1 = ls.index('.')
    s = ls[:in1]
    ls = ls[(in1 + 1):]
    numbers.insert(x, s)
    
flag1 = False
b = df2['estado']
for x in range(68):
    if flag1:
        break
    else:
        flag1 = True
        ss = np.where(b == estado_iniciales)
        estado_iniciales_string = str(ss)
        index_b1 = estado_iniciales_string.index('[')
        index_b2 = estado_iniciales_string.index(']')
        estado_iniciales_valor = estado_iniciales_string[(index_b1 + 1):index_b2]'
"""
df.fillna(0, inplace = True)
df.replace({"pe" : 69}, inplace = True)
for x in range(68):
    estado = df2.iloc[x][0]
    valor = (df2.iloc[x][1] + 1)
    df.replace({estado : valor}, inplace = True)
    
print(df)

# ************************************************************************************
# ************************************************************************************

X = df.iloc[:,1:] # independent variables
y = df.nombre # dependent variable
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# K Means Cluster
model = KMeans(n_clusters=3, random_state=11)
model.fit(X)
print (model.labels_)

df['e1'] = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

# Set the size of the plot
plt.figure(figsize=(10,7))

# Create a colormap
colormap = np.array(['red', 'blue', 'green'])

# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(df['e1'], df['e2'],
c=colormap[df.e1], marker='o', s=50)
plt.xlabel('e1')
plt.ylabel('e2')
plt.title('1')

plt.subplot(2, 2, 2)
plt.scatter(df['e1'], df['e2'],
c=colormap[df.e1], marker='o', s=50)
plt.xlabel('e1')
plt.ylabel('e2')
plt.title('2')

plt.subplot(2, 2, 3)
plt.scatter(df['e1'], df['e2'],
c=colormap[df.e1],marker='o', s=50)
plt.xlabel('e1')
plt.ylabel('e2')
plt.title('3')

plt.subplot(2, 2, 4)
plt.scatter(df['e1'], df['e1'],
c=colormap[df.e1],marker='o', s=50)
plt.xlabel('e1')
plt.ylabel('e2')
plt.title('4')
plt.tight_layout()

K = range(1,9)
KM = [KMeans(n_clusters=k).fit(X) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss-wcss
varExplained = bss/tss*100

kIdx = 10-1
##### plot ###
kIdx = 2

# elbow curve
# Set the size of the plot
plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

plt.subplot(1, 2, 2)
plt.plot(K, varExplained, 'b*-')
plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)

plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.tight_layout()

# ************************************************************************************
# ************************************************************************************