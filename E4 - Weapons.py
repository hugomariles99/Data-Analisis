from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
from scipy.cluster import hierarchy

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

    """
    Tier:
        'Legendary'
        'Exotic'
    	 [1 0]

    Type:
        'Trace Rifle'
        'Sword'
        'Submachine Gun'
        'Sniper Rifle'
        'Sidearm'
        'Shotgun'
        'Scout Rifle'
        'Rocket Launcher'
        'Pulse Rifle'
        'Machine Gun'
        'Linear Fusion Rifle'
        'Hand Cannon'
        'Grenade Launcher'
        'Glaive'
        'Fusion Rifle'
        'Combat Bow'
        'Auto Rifle'
    	 [16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0]

    Category:
        'Power'
        'KineticSlot'
        'Energy'
    	 [2 1 0]

    Element:
        'Void'
        'Stasis'
        'Solar'
        'Kinetic'
        'Arc'
    	 [4 3 2 1 0]
    """

    # ******************* Matriz de correlacion ******************* #
    plt.rcParams.update({'font.size': 8})

    columnas = ['Recoil',
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
                'Category']

    plt.figure(figsize=(16, 12))
    sns.set(style="ticks",
            color_codes=True)
    sns.pairplot(datasetWeapons[columnas],
                 corner=True,
                 hue="Category",
                 palette='flare')

    mdc = "Marriz de correlacion - Weapons"
    plt.title(mdc)
    plt.savefig(mdc + ".pdf")

   # ******************* Heatmap ******************* #
    datasetWeaponsHeatmap = datasetWeapons.drop([
    ],
        axis=1
    )
    corr = datasetWeaponsHeatmap.corr(method='pearson')

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

    mdc = "Heatmap - Weapons"
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

    X = datasetWeapons  # independent variables
    y = datasetWeapons['Category']   # dependent variable

    x_plot = 'Stability'
    x_plot2 = 'Stability'
    y_plot = 'Recoil'
    y_plot2 = 'AA'

    # *********************************** K = 6 ***********************************
    a = np.arange(datasetWeapons.shape[0])
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

    datasetWeapons['Model Labels'] = np.choose(model.labels_,
                                               [0, 1, 2]).astype(np.int64)

    print("Accuracy :",
          metrics.accuracy_score(datasetWeapons['Category'],
                                 datasetWeapons['Model Labels'])
          )

    print("Classification report :",
          metrics.classification_report(datasetWeapons['Category'],
                                        datasetWeapons['Model Labels'])
          )

    # ----------------------------------- #
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    # ----------------------------------- #

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

    title1 = "Reported - " + x_plot + " vs " + y_plot
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

    title2 = "Predicted - " + x_plot + " vs " + y_plot
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

    title3 = "Reported - " + x_plot2 + " vs " + y_plot2
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

    title4 = "Predicted - " + x_plot2 + " vs " + y_plot2
    ax.fig.suptitle(title4, x=0.5, y=1)

    for lh in ax._legend.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [50]
    #leg._loc = 2
    #leg.set_bbox_to_anchor([0.9, 0.9])

    plt.savefig(title4 + ".png")

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
    ep = "Elbowpoint - Weapons"
    plt.title(ep)
    plt.savefig(ep + ".pdf")
    plt.tight_layout()

    # *********************************** CJA ***********************************

    columnasCJA = ['Type',
                   'Year',
                   'Season',
                   'Category']

    plt.figure(figsize=(16, 12))
    Z = hierarchy.linkage(datasetWeapons[columnasCJA], 'single')

    plt.figure(figsize=(16, 12))
    dn = hierarchy.dendrogram(Z,
                              get_leaves=True,
                              distance_sort=True,
                              show_leaf_counts=True,
                              leaf_font_size=2,
                              truncate_mode='none')
    plt.savefig('CJA Full - Weapons.pdf')

    plt.figure(figsize=(16, 12))
    dn = hierarchy.dendrogram(Z,
                              get_leaves=True,
                              distance_sort=True,
                              show_leaf_counts=True,
                              leaf_font_size=2,
                              truncate_mode='level')
    plt.savefig('CJA Simplified - Weapons.pdf')

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
