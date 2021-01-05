import pandas as pd 
import copy
import numpy as np 
import sklearn
from sklearn.cluster import MeanShift,KMeans, estimate_bandwidth
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm
#from code.data import load_data

def clustering(data):
    #coordinate = data.as_matrix(columns = ['latitude','longitude'])
    coordinate = data[['latitude','longitude']].to_numpy()
    bandwidth = estimate_bandwidth(coordinate, quantile = 0.02)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(coordinate)
    labels = meanshift.labels_
    cluster_centers = meanshift.cluster_centers_
    n_clusters_ = len(np.unique(labels))
    data['cluster_grp'] = np.nan
    for i in range(len(coordinate)):
        data['cluster_grp'].iloc[i] = labels[i]
    print(n_clusters_)
    print(data.head(100))
    
    #plotting data after clustering
    '''
    colors = 10*['r','g','b','c','k','y','m']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(coordinate)):
        ax.scatter(coordinate[i][0], coordinate[i][1], c=colors[labels[i]], marker='o')

    ax.scatter(cluster_centers[:,0],cluster_centers[:,1],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

    plt.show()
    '''
    return cluster_centers

data = pd.read_csv("Dataset_US.csv")
#data = data.head(500)
clusters = clustering(data)
coordinates = data[['latitude','longitude']].to_numpy()
plt.figure(figsize=(5,5))
plt.scatter(clusters[:,1], clusters[:,0], c='#99cc99', edgecolor='None', alpha=0.7, s=40)
plt.scatter(coordinates[:,1], coordinates[:,0], c='mediumvioletred', alpha=0.2, s=1) #c='k' for black
plt.grid('off')
plt.axis('off')
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().autoscale_view('tight')
plt.show()
print("yes")

'''
def density_map(latitudes, longitudes, center, bins=1000, radius=0.1):  
    #cmap = matplotlib.cm.get_cmap('Reds') #copy.copy(matplotlib.cm.jet)
    #cmap.set_bad((0,0,0))  # Fill background with black

    # Center the map around the provided center coordinates
    histogram_range = [
        [center[1] - radius , center[1] + radius],
        [center[0] - radius, center[0] + radius]
    ]
    
    fig = plt.figure(figsize=(5,5))
    plt.hist2d(longitudes, latitudes, bins=bins, norm=LogNorm(),
               cmap=plt.cm.gray, range=histogram_range) '
    plt.hist2d(longitudes, latitudes, bins=bins, norm=LogNorm(), cmap = matplotlib.cm.get_cmap('Reds'), range=histogram_range) 
    # Remove all axes and annotations to keep the map clean and simple
    print("yes")
    plt.grid('off')
    plt.axis('off')
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
    print("yes")
usa = [37.0902, 95.7129]
data = data[(data.latitude != 0.00) & (data.longitude != 0.00)]
coordinates = data[['latitude','longitude']].to_numpy()
density_map(coordinates[:,0], coordinates[:,1], center = usa) '''