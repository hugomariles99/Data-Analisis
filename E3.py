import pandas as pd
import pydot as pt
from sklearn import tree
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing

train_data = pd.read_csv("Datasets/Examen Unidad 3/Train.csv")
test_data = pd.read_csv("Datasets/Examen Unidad 3/Test.csv")
train_data_original = train_data

pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)

print(train_data)
print()
print(test_data)

# print(train_data.describe())
print()
# print(train_data.describe(include = 'all')); print()
# print(train_data); print()

"""
print("Test - Accuracy :",
      metrics.accuracy_score(
          y_test,
          clf.predict(x_test)))

print("Test - Confusion matrix :\n",
      metrics.confusion_matrix(
          y_test,
          clf.predict(x_test)))

print("Test - classification report :\n",
      metrics.classification_report(
          y_test,
          clf.predict(x_test)))
"""
