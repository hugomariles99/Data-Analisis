#Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


""" *************************** IMPORTAR CSV *************************** """
df = pd.read_csv("Datasets/vHoneyNeonic_v03_para_knn_800_datos.csv")


""" Descripcion general del dataset"""
#print(df.head(10))
#print(df.describe())
#print(df.info())


""" Histograma para cada columna """
#df.hist()
#plt.show()


""" *************************** GRAFICAS DE BARRAS *************************** """
""" Grafica de las columnas de interes """
#sb.factorplot('totalprod',data=df,kind="count", aspect=3)
#sb.factorplot('nAllNeonic',data=df,kind="count", aspect=3)
#sb.factorplot('numcol',data=df,kind="count", height=8, aspect=16)
#sb.factorplot('Region', data=df,kind="count", aspect=3)


""" *************************** GRAFICA DE PUNTOS *************************** """
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 12)
plt.style.use('ggplot')


""" Etiquetamos las siguientes columnas """
df['totalprod']=LabelEncoder().fit_transform(df['totalprod'])
df['nAllNeonic']=LabelEncoder().fit_transform(df['nAllNeonic'])
df['numcol']=LabelEncoder().fit_transform(df['numcol'])
df['Region']=LabelEncoder().fit_transform(df['Region'])


""" Calcular # de registros para cada estado """
y = df['Etiquetas'].values
print(df.groupby('Etiquetas').size())
print()


""" Entradas para el algoritmo """
X = df[['totalprod','nAllNeonic']].values 


""" Sets de entrenamiento y test """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


""" Escalado de características """
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


""" Valor de k """
n_neighbors = 3


""" Algoritmo knn """
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)


print("\n")
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
print("\n")


#Confirmar presicion del modelo
pred = knn.predict(X_test)


#Datos de test
print("*****Datos reales*****")
print(y_test)
print("\n")
#Predicciones del algoritmo
print("*****Prediciones del algoritmo***")
print(pred)
print("\n")


#Evaluar el modelo
print("*****Matriz de Confusion*****")
print(confusion_matrix(y_test, pred))
print("\n")
print("*****Reporte de evalucion del modelo*****")
print(classification_report(y_test, pred))



h = .2  #pasos en la grafica


# Crea el mapa de colore
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff'])
                            

#creamos una instancia del clasificador de vecinos y ajustamos los datos
clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)
 

#Trace el límite de decisión. Para eso, asignaremos un color a cada
# punto en la grafica[x_min, x_max]x[y_min, y_max]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 

# Colorcar el resultado en la grafica de color.
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 

# Trazar los puntos de entreanmiento
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    

#Definir colores para cada etiqueta
patch0 = mpatches.Patch(color='#FF0000', label='a')
patch1 = mpatches.Patch(color='#ff9933', label='b')
patch2 = mpatches.Patch(color='#FFFF00', label='c')
patch3 = mpatches.Patch(color='#00ffff', label='d')
plt.legend(handles=[patch0, patch1, patch2, patch3])
plt.title("Class classification (k = 6)")
              #% (n_neighbors, weights))
plt.show()

