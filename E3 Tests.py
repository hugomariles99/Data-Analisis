import pandas as pd
import pydot as pt
from io import StringIO
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


train_data = pd.read_csv("Datasets/Examen Unidad 3/Train.csv")
test_data = pd.read_csv("Datasets/Examen Unidad 3/Test.csv")

pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)
#
#
#
#
#
#
#
#
#
#

""" *********************** Choose column *********************** """
# ----------------------------------------------------------------- #
w = 'Profession'
# ----------------------------------------------------------------- #
""" *********************** Choose column *********************** """

#
#
#
#
#
#
#
#
#
#
# ************************************************************************************************ #
""" ******************************************************************************************** """
# ************************************************************************************************ #
#                                            Preprocessing                                         #
# ------------------------------------------------------------------------------------------------ #
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

test_data['Work_Experience'] = test_data['Work_Experience'].fillna(15)
test_data['Family_Size'] = test_data['Family_Size'].fillna(9)

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
labelEncoder_sort(test_data)

train_data.sort_values(
    by='ID',
    inplace=True)

test_data.sort_values(
    by='ID',
    inplace=True)

# ------------------------------------------------------------------------------------------------ #
#                                            Preprocessing                                         #
# ************************************************************************************************ #
""" ******************************************************************************************** """
# ************************************************************************************************ #
#
#
#
#
#
#
#
#
#
#

#
#
#
#
#
#
#
#
#
#
# ************************************************************************************************ #
""" ******************************************************************************************** """
# ************************************************************************************************ #
#                                               Processing                                         #
# ------------------------------------------------------------------------------------------------ #

""" Initiate Variables """
x_trd_data = train_data
y_trd_data = train_data[w]

sc = StandardScaler()
sc.fit(x_trd_data)
x_trd_data = sc.transform(x_trd_data)

x_trd_train, x_trd_test, y_trd_train, y_trd_test = train_test_split(x_trd_data,
                                                                    y_trd_data,
                                                                    test_size=0.9,
                                                                    random_state=0)

x_ted_data = test_data
y_ted_data = test_data[w]

sc = StandardScaler()
sc.fit(x_ted_data)
x_ted_data = sc.transform(x_ted_data)

x_ted_train, x_ted_test, y_ted_train, y_ted_test = train_test_split(x_ted_data,
                                                                    y_ted_data,
                                                                    test_size=0.9,
                                                                    random_state=0)

""" *********************** Train Dataset *********************** """
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x_trd_train, y_trd_train)

# ---------- Accuracy ---------- #
trd_accuracy = metrics.accuracy_score(y_trd_train,
                                      clf.predict(x_trd_train))

# ---------- Confusion Matrix ---------- #
predictions = clf.predict(x_trd_test)
cm = confusion_matrix(y_trd_test,
                      predictions,
                      labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)

disp.plot()
plt.title('Train Dataset / Accuracy = ' + str(trd_accuracy))
plt.show()

trd_train_report = metrics.classification_report(y_trd_train, clf.predict(x_trd_train))
print(trd_train_report)
trd_train_report = trd_train_report[:-150]

trd_test_report = metrics.classification_report(y_trd_test, clf.predict(x_trd_test))
print(trd_test_report)
trd_test_report = trd_test_report[:-150]

""" *********************** Test Dataset *********************** """
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x_ted_train, y_ted_train)

# ---------- Accuracy ---------- #
ted_accuracy = metrics.accuracy_score(y_ted_train,
                                      clf.predict(x_ted_train))

# ---------- Confusion Matrix ---------- #
predictions = clf.predict(x_ted_test)
cm = confusion_matrix(y_ted_test,
                      predictions,
                      labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)

disp.plot()
plt.title('Test Dataset / Accuracy = ' + str(ted_accuracy))
plt.show()

ted_train_report = metrics.classification_report(y_ted_train, clf.predict(x_ted_train))
print(ted_train_report)
ted_train_report = ted_train_report[:-150]

ted_test_report = metrics.classification_report(y_ted_test, clf.predict(x_ted_test))
print(ted_test_report)
ted_test_report = ted_test_report[:-150]

""" *********************** Tree *********************** """
# tree.export_graphviz(clf, out_file='tree.dot')

out_data = StringIO()
tree.export_graphviz(clf,
                     out_file=out_data,
                     class_names=clf.classes_.astype(int).astype(str),
                     feature_names=(
                         'ID',
                         'Gender',
                         'Ever_Married',
                         'Age',
                         'Graduated',
                         'Profession',
                         'Work_Experience',
                         'Spending_Score',
                         'Family_Size',
                         'Var_1',
                         'Segmentation'),
                     filled=True,
                     rounded=True,
                     special_characters=True,
                     node_ids=1)
graph = pt.graph_from_dot_data(out_data.getvalue())

out_name = w + "_tree.pdf"
graph[0].write_pdf(out_name)

# ------------------------------------------------------------------------------------------------ #
#                                               Processing                                         #
# ************************************************************************************************ #
""" ******************************************************************************************** """
# ************************************************************************************************ #
#
#
#
#
#
#
#
#
#
#

#
#
#
#
#
#
#
#
#
#
# ************************************************************************************************ #
""" ******************************************************************************************** """
# ************************************************************************************************ #
# ************************************************************************************************ #
#                                        Acuraccy score table                                      #
# ------------------------------------------------------------------------------------------------ #

"""
https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report 
by: Franck Dernoncourt
"""


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC,
            title,
            xlabel,
            ylabel,
            xticklabels,
            yticklabels,
            figure_width=40,
            figure_height=20,
            correct_orientation=False,
            cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    # fig.set_size_inches(cm2inch(40, 20))
    # fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title, cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        # print(v)
        plotMat.append(v)

    # print('plotMat: {0}'.format(plotMat))
    # print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels,
            figure_width, figure_height, correct_orientation, cmap=cmap)


def main():
    plot_classification_report(trd_train_report, 'Train Data - Train Report')
    plt.savefig('Train Data - Train Repo.png', dpi=200, format='png', bbox_inches='tight')

    plot_classification_report(trd_test_report, 'Train Data - Test Report')
    plt.savefig('Train Data - Test Repo.png', dpi=200, format='png', bbox_inches='tight')

    plot_classification_report(ted_train_report, 'Test Data - Train Report')
    plt.savefig('Test Data - Train Repo.png', dpi=200, format='png', bbox_inches='tight')

    plot_classification_report(ted_test_report, 'Test Data - Test Report')
    plt.savefig('Test Data - Test Repo.png', dpi=200, format='png', bbox_inches='tight')
    plot_classification_report(ted_test_report, "")
    plt.close()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------ #
#                                        Acuraccy score table                                      #
# ************************************************************************************************ #
""" ******************************************************************************************** """
# ************************************************************************************************ #
