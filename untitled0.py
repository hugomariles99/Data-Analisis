from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
from scipy.cluster import hierarchy
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import Divider, Size

destinyWeaponsOS = pd.read_csv("Datasets/destinyWeaponsOS.csv")
destinyWeaponsHM = pd.read_csv("Datasets/destinyWeaponsHM.csv")

destinyWeaponsHM = destinyWeaponsHM.drop(columns=['Perks 15',
                                                  'Perks 16'
                                                  ])

destinyWeaponsAll = pd.concat([destinyWeaponsOS,
                               destinyWeaponsHM],
                              axis=0)


def stuff(data):
    #
    #
    #
    # ********************************************************* #
    """ ******************* PREPROCESSING ******************* """
    # --------------------------------------------------------- #

    # ******************* Remove columns ******************* #
    datasetWeapons2 = data.drop(columns=['Hash',
                                         'Id',
                                         'Tag',
                                         'Source',
                                         'Power',
                                         'Power Limit',
                                         'Masterwork Type',
                                         'Masterwork Tier',
                                         'Owner',
                                         'Locked',
                                         'Equipped',
                                         'Event',
                                         'Notes',
                                         'Equip',
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
                                         'Perks 11',
                                         'Perks 12',
                                         'Perks 13',
                                         'Perks 14',
                                         ])

    datasetWeapons = data.drop(columns=['Name',
                                        'Hash',
                                        'Id',
                                        'Tag',
                                        'Source',
                                        'Power',
                                        'Power Limit',
                                        'Masterwork Type',
                                        'Masterwork Tier',
                                        'Owner',
                                        'Locked',
                                        'Equipped',
                                        'Event',
                                        'Notes',
                                        'Equip',
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
                                        'Perks 11',
                                        'Perks 12',
                                        'Perks 13',
                                        'Perks 14',
                                        ])

    print(datasetWeapons)

    # ******************* Remove 'Rare' Tier ******************* #
    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Tier'] == 'Rare'].index,
        inplace=True)

    # -------------------------- #
    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Trace Rifle'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Sword'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Sniper Rifle'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Sidearm'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Shotgun'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Scout Rifle'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Linear Fusion Rifle'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Hand Cannon'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Grenade Launcher'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Glaive'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Combat Bow'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Fusion Rifle'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Pulse Rifle'].index,
        inplace=True)

    datasetWeapons.drop(
        datasetWeapons.loc[datasetWeapons['Type'] == 'Rocket Launcher'].index,
        inplace=True)

    datasetWeapons.reset_index(drop=True,
                               inplace=True)

    # ******************* Add ID column ******************* #
    datasetWeapons.insert(0,
                          'ID',
                          datasetWeapons.index + 1)

    # ******************* Describes ******************* #
    print(datasetWeapons.describe())
    print()
    print(datasetWeapons.describe(include='all'))

    # ******************* labelEncoder ******************* #
    columnas = ['Tier',
                'Type',
                'Category',
                'Element'
                ]

    le = preprocessing.LabelEncoder()

    def labelEncoder_sort(df):
        for x in columnas:
            datasetWeapons.sort_values(
                by=x,
                ascending=False,
                inplace=True)

            """
            print(x + ":")

            print('{0} {1}'.format(
                '\t',
                df[x].unique()))
            """

            datasetWeapons[x] = le.fit_transform(
                datasetWeapons[x].astype(str))

            """
            print('{0} {1}'.format(
                '\t',
                df[x].unique()))
            print()
            """

    labelEncoder_sort(datasetWeapons)

    datasetWeapons.sort_values(
        by='ID',
        inplace=True)

    x = datasetWeapons  # independent variables
    y = datasetWeapons['Type']   # dependent variable

    x_plot = 'ROF'
    x_plot2 = 'ROF'
    y_plot = 'Impact'
    y_plot2 = 'Reload'

    # *********************************** K = 3 ***********************************
    a = np.arange(datasetWeapons.shape[0])
    colormap = np.array(['#A30000', '#005a80', '#03971B'])
    """
    c=a,
    cmap='cool',
    """

    # K Means Cluster
    model = KMeans(n_clusters=3, random_state=11)
    model.fit(x)
    print(model.labels_)

    plt.rcParams.update({'font.size': 25})

    datasetWeapons['Model Labels'] = np.choose(model.labels_,
                                               [0, 1, 2]).astype(np.int64)

    print("Accuracy :",
          metrics.accuracy_score(datasetWeapons['Type'],
                                 datasetWeapons['Model Labels'])
          )

    print("Classification report :",
          metrics.classification_report(datasetWeapons['Type'],
                                        datasetWeapons['Model Labels'])
          )

    # ----------------------------------- #

    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500

    sns.set(font_scale=1)
    ax = sns.relplot(data=datasetWeapons,
                     x=x_plot,
                     y=y_plot,
                     hue="Category",
                     style="Category",
                     s=250,
                     alpha=1,
                     edgecolor='black',
                     aspect=11.7/8.27)

    title1 = ""
    ax.fig.suptitle(title1, x=0.5, y=1)

    for lh in ax._legend.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [50]
    #leg._loc = 2
    #leg.set_bbox_to_anchor([0.9, 0.9])

    plt.savefig(title1 + ".png")

    # ----------------------------------- #

    sns.set(font_scale=1)
    ax = sns.relplot(data=datasetWeapons,
                     x=x_plot,
                     y=y_plot,
                     hue="Model Labels",
                     style="Model Labels",
                     s=250,
                     alpha=1,
                     edgecolor='black',
                     aspect=11.7/8.27)

    title2 = ""
    ax.fig.suptitle(title2, x=0.5, y=1)

    for lh in ax._legend.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [50]
    #leg._loc = 2
    #leg.set_bbox_to_anchor([0.9, 0.9])

    plt.savefig(title2 + ".png")
    # ----------------------------------- #

    sns.set(font_scale=1)
    ax = sns.relplot(data=datasetWeapons,
                     x=x_plot2,
                     y=y_plot2,
                     hue="Category",
                     style="Category",
                     s=250,
                     alpha=1,
                     edgecolor='black',
                     aspect=11.7/8.27)

    title3 = ""
    ax.fig.suptitle(title3, x=0.5, y=1)

    for lh in ax._legend.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [50]
    #leg._loc = 2
    #leg.set_bbox_to_anchor([0.9, 0.9])

    plt.savefig(title3 + ".png")
    # ----------------------------------- #

    sns.set(font_scale=1)
    ax = sns.relplot(data=datasetWeapons,
                     x=x_plot2,
                     y=y_plot2,
                     hue="Model Labels",
                     style="Model Labels",
                     s=250,
                     alpha=1,
                     edgecolor='black',
                     aspect=11.7/8.27)

    title4 = ""
    ax.fig.suptitle(title4, x=0.5, y=1)

    for lh in ax._legend.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [50]
    #leg._loc = 2
    #leg.set_bbox_to_anchor([0.9, 0.9])

    plt.savefig(title4 + ".png")
    # ----------------------------------- #

    datasetWeapons2 = datasetWeapons2.drop(columns=['Charge Time',
                                                    'Draw Time',
                                                    'Accuracy',
                                                    'Velocity',
                                                    'Blast Radius'])
    w = datasetWeapons2.loc[(datasetWeapons2['ROF'] == 900)]
    print(w)

    """
    'ID',
    'Tier',
    'Type',
    'Category',
    'Element',
    'Year',
    'Season',
            'Recoil',
            'AA',
            'Impact',
            'Range',
            'Zoom',
            'Blast Radius',
            'Velocity',
            'Stability',
            'ROF',
            'Reload',
            'Mag',
            'Charge Time',
            'Draw Time',
            'Accuracy'
    """


stuff(destinyWeaponsAll)
