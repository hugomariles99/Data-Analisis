from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
from scipy.cluster import hierarchy

destinyArmorOS = pd.read_csv("Datasets/destinyArmorOS.csv")
destinyArmorHM = pd.read_csv("Datasets/destinyArmorHM.csv")

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)

destinyArmorHM = destinyArmorHM.drop(columns=['Perks 11',
                                              'Perks 12',
                                              'Perks 13',
                                              'Perks 14',
                                              'Perks 15',
                                              ])

destinyArmorAll = pd.concat([destinyArmorOS,
                             destinyArmorHM],
                            axis=0)


def stuff(data):
    #
    #
    #
    # ********************************************************* #
    """ ******************* PREPROCESSING ******************* """
    # --------------------------------------------------------- #

    # ******************* Remove columns ******************* #
    datasetArmor = data.drop(columns=['Name',
                                      'Hash',
                                      'Id',
                                      'Tag',
                                      'Source',
                                      'Power',
                                      'Power Limit',
                                      'Masterwork Tier',
                                      'Owner',
                                      'Armor2.0',
                                      'Locked',
                                      'Equipped',
                                      'Event',
                                      'Custom',
                                      'Custom (Base)',
                                      'Seasonal Mod',
                                      'Notes',
                                      'Perks 0',
                                      'Perks 1',
                                      'Perks 2',
                                      'Perks 3',
                                      'Perks 4',
                                      'Perks 5',
                                      'Perks 6',
                                      'Perks 7',
                                      'Perks 8',
                                      'Perks 9',
                                      'Perks 10',
                                      ])

    print(datasetArmor)

    # ******************* Remove 'Rare' Tier ******************* #
    datasetArmor.drop(
        datasetArmor.loc[datasetArmor['Tier'] == 'Rare'].index,
        inplace=True)

    datasetArmor.reset_index(drop=True,
                             inplace=True)

    # ******************* Add ID column ******************* #
    datasetArmor.insert(0,
                        'ID',
                        datasetArmor.index + 1)

    # ******************* Describes ******************* #
    print(datasetArmor.describe())
    print()
    print(datasetArmor.describe(include='all'))

    # ******************* Add labels Stats ******************* #
    """
    STAT
    0 * -8
    1 * 9-16
    2 * 17-24
    3 * 25-32
    4 * 33-40
    5 * 41+
    """

    stats = ['Mobility',
             'Resilience',
             'Recovery',
             'Discipline',
             'Intellect',
             'Strength'
             ]

    for z in stats:
        datasetSize = datasetArmor.shape[0]
        total = []

        for w in range(datasetSize):
            x = datasetArmor.iloc[w][z]

            label = 0
            if x <= 8:
                label = 0

            elif x >= 9 and x <= 16:
                label = 1

            elif x >= 17 and x <= 24:
                label = 2

            elif x >= 25 and x <= 32:
                label = 3

            elif x >= 33 and x <= 40:
                label = 4

            elif x >= 41:
                label = 5
            # print(label)
            total.append(label)

        columnName = z + " Label"
        # print(total)
        datasetArmor.insert(datasetArmor.shape[1],
                            columnName,
                            total)

    """
    Total
    0 * -40
    1 * 41-55
    2 * 56-70
    3 * 71-85
    4 * 86-100
    5 * 101+
    """
    datasetSize = datasetArmor.shape[0]
    totalF = []

    for w in range(datasetSize):
        x = datasetArmor.iloc[w]['Total']

        label = 0
        if x <= 40:
            label = 0

        elif x >= 41 and x <= 55:
            label = 1

        elif x >= 56 and x <= 70:
            label = 2

        elif x >= 71 and x <= 85:
            label = 3

        elif x >= 86 and x <= 100:
            label = 4

        elif x >= 101:
            label = 5
            # print(label)
        totalF.append(label)

    datasetArmor.insert(datasetArmor.shape[1],
                        'Total Label',
                        totalF)

    """
    STAT (Base)
    0 * -5
    1 * 6-10
    2 * 11-15
    3 * 16-20
    4 * 21-25
    5 * 26+
    """
    stats = ['Mobility',
             'Resilience',
             'Recovery',
             'Discipline',
             'Intellect',
             'Strength'
             ]
    for z in stats:
        datasetSize = datasetArmor.shape[0]
        total = []

        for w in range(datasetSize):
            x = datasetArmor.iloc[w][z + ' (Base)']

            label = 0
            if x <= 5:
                label = 0

            elif x >= 6 and x <= 10:
                label = 1

            elif x >= 11 and x <= 15:
                label = 2

            elif x >= 16 and x <= 20:
                label = 3

            elif x >= 21 and x <= 25:
                label = 4

            elif x >= 26:
                label = 5
                # print(label)
            total.append(label)

        columnName = z + " (Base) Label"
        # print(total)
        datasetArmor.insert(datasetArmor.shape[1],
                            columnName,
                            total)

    """
    Total (Base)
    0 * -40
    1 * 41-46
    2 * 47-52
    3 * 53-58
    4 * 59-64
    5 * 65+
    """
    datasetSize = datasetArmor.shape[0]
    total = []

    for w in range(datasetSize):
        x = datasetArmor.iloc[w]['Total (Base)']

        label = 0
        if x <= 40:
            label = 0

        elif x >= 41 and x <= 46:
            label = 1

        elif x >= 47 and x <= 52:
            label = 2

        elif x >= 53 and x <= 58:
            label = 3

        elif x >= 59 and x <= 64:
            label = 4

        elif x >= 65:
            label = 5
        # print(label)
        total.append(label)

    datasetArmor.insert(datasetArmor.shape[1],
                        'Total (Base) Label',
                        total)

    # ******************* labelEncoder ******************* #
    columnas = ['Tier',
                'Type',
                'Equippable',
                'Masterwork Type'
                ]

    le = preprocessing.LabelEncoder()

    def labelEncoder_sort(df):
        for x in columnas:
            datasetArmor.sort_values(
                by=x,
                ascending=False,
                inplace=True)

            """
            print(x + ":")

            print('{0} {1}'.format(
                '\t',
                df[x].unique()))
            """

            datasetArmor[x] = le.fit_transform(
                datasetArmor[x].astype(str))

            """
            print('{0} {1}'.format(
                '\t',
                df[x].unique()))
            print()
            """
    """
    Tier:
        'Legendary'
        'Exotic'
    	 [1 0]

    Type:
        'Warlock Bond' 
        'Titan Mark'
        'Leg Armor' 
        'Hunter Cloak'
        'Helmet'
        'Gauntlets' 
        'Chest Armor'
    	 [6 5 4 3 2 1 0]

    Equippable:
    	 'Warlock' 
         'Titan' 
         'Hunter'
    	 [2 1 0]

    Masterwork Type:
    	 'Void Energy Capacity'
         'Stasis Cost' 
         'Solar Energy Capacity'
         'Arc Energy Capacity'
    	 [3 2 1 0]
    """

    labelEncoder_sort(datasetArmor)

    datasetArmor.sort_values(
        by='ID',
        inplace=True)

    # print(datasetArmor)

    # ******************* Matriz de correlacion ******************* #
    # columnas = ['column_nd', 'column_nc']
    plt.rcParams.update({'font.size': 8})

    columnas = ['Mobility',
                'Resilience',
                'Recovery',
                'Discipline',
                'Intellect',
                'Strength',
                'Total Label', ]

    plt.figure(figsize=(16, 12))
    sns.set(style="ticks",
            color_codes=True)
    sns.pairplot(datasetArmor[columnas],
                 corner=True,
                 hue="Total Label",
                 palette='Spectral')

    mdc = "Marriz de correlacion - Armor"
    plt.title(mdc)
    plt.savefig(mdc + ".pdf")

    # ******************* Heatmap ******************* #
    datasetArmorHeatmap = datasetArmor.drop(['Mobility Label',
                                             'Resilience Label',
                                             'Recovery Label',
                                             'Discipline Label',
                                             'Intellect Label',
                                             'Strength Label',
                                             'Total Label',
                                             'Mobility (Base) Label',
                                             'Resilience (Base) Label',
                                             'Recovery (Base) Label',
                                             'Discipline (Base) Label',
                                             'Intellect (Base) Label',
                                             'Strength (Base) Label',
                                             'Total (Base) Label'
                                             ],
                                            axis=1
                                            )
    corr = datasetArmorHeatmap.corr(method='pearson')

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(16, 12))
        ax = sns.heatmap(corr,
                         mask=mask,
                         annot=True,
                         fmt='.1f',
                         linewidths=.5,
                         cmap="gist_heat",
                         vmin=0.0,
                         annot_kws={"size": 16})

    mdc = "Heatmap - Armor"
    plt.title(mdc)
    plt.savefig(mdc + ".pdf")
    # --------------------------------------------------------- #
    """ ******************* PREPROCESSING ******************* """
    # ********************************************************* #
    #
    #
    #

    #
    #
    #
    # ********************************************************* #
    """ ********************* PROCESSING ******************** """
    # --------------------------------------------------------- #

    X = datasetArmor  # independent variables
    y = datasetArmor['Equippable']   # dependent variable

    x_plot = 'Total'
    x_plot2 = 'Total (Base)'
    y_plot = 'Mobility'
    y_plot2 = 'Mobility (Base)'

    # *********************************** K = 6 ***********************************
    a = np.arange(datasetArmor.shape[0])
    colormap = np.array(['#A30000', '#005a80', '#03971B'])
    """
    c=a,
    cmap='cool',
    """

    # K Means Cluster
    model = KMeans(n_clusters=3, random_state=11)
    model.fit(X)
    print(model.labels_)

    plt.rcParams.update({'font.size': 25})

    datasetArmor['Model Labels'] = np.choose(model.labels_,
                                             [0, 1, 2]).astype(np.int64)

    print("Accuracy :",
          metrics.accuracy_score(datasetArmor['Equippable'],
                                 datasetArmor['Model Labels'])
          )

    print("Classification report :",
          metrics.classification_report(datasetArmor['Equippable'],
                                        datasetArmor['Model Labels'])
          )

    # ----------------------------------- #
    plt.figure(figsize=(16, 12))
    plt.scatter(datasetArmor[x_plot],
                datasetArmor[y_plot],
                c=colormap[datasetArmor['Equippable']],
                marker='.',
                s=4000,
                alpha=0.3)
    plt.xlabel(x_plot)
    plt.ylabel(y_plot)
    name1 = "Equippable - " + x_plot + " vs " + y_plot
    plt.title(name1)
    plt.savefig(name1 + ".pdf")

    # ----------------------------------- #
    plt.figure(figsize=(16, 12))
    plt.scatter(datasetArmor[x_plot],
                datasetArmor[y_plot],
                c=colormap[datasetArmor['Model Labels']],
                marker='.',
                s=4000,
                alpha=0.3)
    plt.xlabel(x_plot)
    plt.ylabel(y_plot)
    name2 = "Predicted - " + x_plot + " vs " + y_plot
    plt.title(name2)
    plt.savefig(name2 + ".pdf")

    # ----------------------------------- #
    plt.figure(figsize=(16, 12))
    plt.scatter(datasetArmor[x_plot2],
                datasetArmor[y_plot2],
                c=colormap[datasetArmor['Equippable']],
                marker='.',
                s=4000,
                alpha=0.3)
    plt.xlabel(x_plot2)
    plt.ylabel(y_plot2)
    name3 = "Equippable - " + x_plot2 + " vs " + y_plot2
    plt.title(name3)
    plt.savefig(name3 + ".pdf")

    # ----------------------------------- #
    plt.figure(figsize=(16, 12))
    plt.scatter(datasetArmor[x_plot2],
                datasetArmor[y_plot2],
                c=colormap[datasetArmor['Model Labels']],
                marker='.',
                s=4000,
                alpha=0.3)
    plt.xlabel(x_plot2)
    plt.ylabel(y_plot2)
    name4 = "Predicted - " + x_plot2 + " vs " + y_plot2
    plt.title(name4)
    plt.savefig(name4 + ".pdf")

    plt.show()

    # *********************************** Elbowpoint ***********************************
    K = range(1, 9)
    KM = [KMeans(n_clusters=k).fit(X) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
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
    plt.figure(figsize=(16, 12))

    plt.plot(K,
             varExplained,
             'b*-')

    plt.plot(K[kIdx],
             varExplained[kIdx],
             marker='o',
             markersize=12,
             markeredgewidth=2,
             markeredgecolor='r',
             markerfacecolor='None')
    plt.grid(True)

    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    ep = "Elbowpoint - Armor"
    plt.title(ep)
    plt.savefig(ep + ".pdf")
    plt.tight_layout()

    # *********************************** CJA ***********************************

    columnasCJA = ['Type',
                   'Year',
                   'Season',
                   'Total Label']

    plt.figure(figsize=(16, 12))
    Z = hierarchy.linkage(datasetArmor[columnasCJA], 'single')
    dn = hierarchy.dendrogram(Z,
                              get_leaves=True,
                              count_sort=True,
                              show_leaf_counts=True,
                              leaf_font_size=2)
    plt.savefig('CJA - Armor.pdf')


# --------------------------------------------------------- #
""" ********************* PROCESSING ******************** """
# ********************************************************* #
#
#
#
"""
                                         'ID',
                                         'Tier',
                                         'Type',
                                         'Equippable',
                                         'Masterwork Type',
                                         'Year',
                                         'Season',
                                         'Mobility',
                                         'Resilience',
                                         'Recovery',
                                         'Discipline',
                                         'Intellect',
                                         'Strength',
                                         'Total',
                                         'Mobility (Base)',
                                         'Resilience (Base)',
                                         'Recovery (Base)',
                                         'Discipline (Base)',
                                         'Intellect (Base)',
                                         'Strength (Base)',
                                         'Total (Base)',
                                         'Mobility Label',
                                         'Resilience Label',
                                         'Recovery Label',
                                         'Discipline Label',
                                         'Intellect Label',
                                         'Strength Label',
                                         'Total Label',
                                         'Mobility (Base) Label',
                                         'Resilience (Base) Label',
                                         'Recovery (Base) Label',
                                         'Discipline (Base) Label',
                                         'Intellect (Base) Label',
                                         'Strength (Base) Label',
                                         'Total (Base) Label'
"""

stuff(destinyArmorAll)
