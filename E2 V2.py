import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist

df = pd.read_csv("Datasets/Examen Unidad 2 - Plantas/plants 3.data")
df2 = pd.read_csv("Datasets/Examen Unidad 2 - Plantas/states.txt")

df['id'] = df.index
# *******************************************************************************

df.fillna(0, inplace = True)

df.replace({"pe" : 69}, inplace = True)
df.replace({"gl" : 70}, inplace = True)

for x in range(68):
    estado = df2.iloc[x][0]
    valor = (df2.iloc[x][1] + 1)
    df.replace({estado : valor}, inplace = True)
    
print(df)

# ************************************************************************************

X = df.iloc[:,1:] # independent variables
y = df.id # dependent variable

# Create a colormap
colormap2 = np.array(['red', 'blue'])
colormap3 = np.array(['red', 'blue', 'green'])
colormap4 = np.array(['red', 'blue', 'green', 'yellow'])

x_plot = 'e1'
c_2_1 = 'e2'
c_2_2 = 'e5'
c_2_3 = 'e8'
c_2_4 = 'e11'

# *********************************** K = 2 ***********************************
# K Means Cluster
model2 = KMeans(n_clusters=2)
model2.fit(X)
#print (model.labels_)

df['id'] = np.choose(model2.labels_, [1, 0]).astype(np.int64)

# Set the size of the plot
plt.figure(figsize=(10,7))

# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(df[x_plot], df[c_2_1], 
c=colormap2[df.id], marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_1)

plt.subplot(2, 2, 2)
plt.scatter(df[x_plot], df[c_2_2], 
c=colormap2[df.id], marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_2)

plt.subplot(2, 2, 3)
plt.scatter(df[x_plot], df[c_2_3], 
c=colormap2[df.id],marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_3)

plt.subplot(2, 2, 4)
plt.scatter(df[x_plot], df[c_2_4], 
c=colormap2[df.id],marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_4)

plt.tight_layout()

# *********************************** K = 3 ***********************************
# K Means Cluster
model3 = KMeans(n_clusters=3)
model3.fit(X)
#print (model.labels_)

df['id'] = np.choose(model3.labels_, [1, 0, 2]).astype(np.int64)

# Set the size of the plot
plt.figure(figsize=(10,7))

# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(df[x_plot], df[c_2_1], 
c=colormap3[df.id], marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_1)

plt.subplot(2, 2, 2)
plt.scatter(df[x_plot], df[c_2_2], 
c=colormap3[df.id], marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_2)

plt.subplot(2, 2, 3)
plt.scatter(df[x_plot], df[c_2_3], 
c=colormap3[df.id],marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_3)

plt.subplot(2, 2, 4)
plt.scatter(df[x_plot], df[c_2_4], 
c=colormap3[df.id],marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_4)

plt.tight_layout()

# *********************************** K = 4 ***********************************
# K Means Cluster
model4 = KMeans(n_clusters=4)
model4.fit(X)
#print (model.labels_)

df['id'] = np.choose(model4.labels_, [1, 0, 2, 3]).astype(np.int64)

# Set the size of the plot
plt.figure(figsize=(10,7))

# Plot Sepal
plt.subplot(2, 2, 1)
plt.scatter(df[x_plot], df[c_2_1], 
c=colormap4[df.id], marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_1)

plt.subplot(2, 2, 2)
plt.scatter(df[x_plot], df[c_2_2], 
c=colormap4[df.id], marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_2)

plt.subplot(2, 2, 3)
plt.scatter(df[x_plot], df[c_2_3], 
c=colormap4[df.id],marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_3)

plt.subplot(2, 2, 4)
plt.scatter(df[x_plot], df[c_2_4], 
c=colormap4[df.id],marker=2, s=50)
plt.xlabel(x_plot)
plt.ylabel(c_2_4)

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

plt.subplot(1, 2, 2)
plt.plot(K, varExplained, 'b*-')
plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)

plt.xlabel('Clusters')
plt.ylabel('Valor del estado')
plt.tight_layout()