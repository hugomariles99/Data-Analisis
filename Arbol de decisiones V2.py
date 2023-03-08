import numpy as np
import pandas as pd
import pydot as pt
from sklearn import tree
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing

iris = datasets.load_iris()
train_data = pd.read_csv("Datasets/Examen Unidad 3/Train.csv")

columnas = [
    # 'ID',
    'Gender',
    'Ever_Married',
    # 'Age',
    'Graduated',
    'Profession',
    # 'Work_Experience',
    'Spending_Score',
    # 'Family_Size',
    'Var_1',
    'Segmentation']

# ---------------- Fill NaN ---------------- #
train_data['Work_Experience'] = train_data['Work_Experience'].fillna(15)
train_data['Family_Size'] = train_data['Family_Size'].fillna(9)

le = preprocessing.LabelEncoder()


def labelEncoder_sort(df):
    for x in columnas:
        df.sort_values(
            by=x,
            ascending=False,
            na_position='first',
            inplace=True)

        """
        print(x + ":")

        print('{0} {1}'.format(
            '\t',
            df[x].unique()))
        """

        df[x] = le.fit_transform(
            df[x].astype(str))

        """
        print('{0} {1}'.format(
            '\t',
            df[x].unique()))
        print()
        """


labelEncoder_sort(train_data)

train_data.sort_values(
    by='ID',
    inplace=True)


# X = iris.data[:, [2, 3]]
X = iris.data
y = iris.target
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
                                                    random_state=0)
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
print("Train - Confusion matrix :", metrics.confusion_matrix(y_train, clf.predict(X_train)))
print("Train - classification report :", metrics.classification_report(y_train, clf.predict(X_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
print("Test - Confusion matrix :", metrics.confusion_matrix(y_test, clf.predict(X_test)))
print("Test - classification report :", metrics.classification_report(y_test, clf.predict(X_test)))
tree.export_graphviz(clf, out_file='tree.dot')


out_data = StringIO()
tree.export_graphviz(clf, out_file=out_data,
                     feature_names=iris.feature_names,
                     class_names=clf.classes_.astype(int).astype(str),
                     filled=True, rounded=True,
                     special_characters=True,
                     node_ids=1,)
graph = pt.graph_from_dot_data(out_data.getvalue())
graph[0].write_pdf("iris.pdf")  # save to pdf
