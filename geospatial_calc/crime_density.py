import os
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely import geometry

#see all panda columns/rows
pd.options.display.max_columns = None
pd.options.display.max_rows = None


''' 
creating a crime density map
    1. silhouette analysis to determine k
    2. k-means algo with initial seeds (possibly taken from the heatmap maxima from the ArcGIS raster)
    3. record all global maxima from k-means
    4. determine some radius, perhaps based off population density
    5. aggregate a property's proximity to each global maxima within the radius
'''


#first determine optimal k for k means
def silhouette_analysis(samples_arr: np.array) -> list:
    '''
    

    Parameters
    ----------
    samples_arr : TYPE
        numpy array of n-dimensions, with potential data point clusters.

    Returns
    -------
    list of n_clusters and their silhouette score; the closer the score to 1, the more optimal.

    '''
    #possible n clusters derived from ArcGIS heatmap peaks for LA county only
    range_n_clusters = [13, 14]
    range_n_clusters_01 = [15, 16, 17]
    
    for n_clusters in range_n_clusters:
        #subplot with 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        
        #1st subplot is the silhouette plot
        ax1.set_xlim([-1, 1])
        
        #(n_clusters+1)*10 to insert blank spaces between silhouette samples
        ax1.set_ylim([0, len(samples_arr) + (n_clusters + 1)*10])
        
        #initialize clusterer with n_clusters value and random seed=10
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(samples_arr)
        
        #silhouette average for all samples
        silhouette_avg = silhouette_score(samples_arr, cluster_labels)
        print('For n_clusters=', n_clusters,
              'The average silhouette score is: ', silhouette_avg)
        
        #calc silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(samples_arr, cluster_labels)
        
        y_lower = 10
        for i in range(n_clusters):
            #aggregate silhouette scores for each sample i, and sort
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i)/n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            
            #label silhouette plots with their respective cluster numbers in the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            #new y_lower for next plot
            y_lower = y_upper + 10
    
        ax1.set_title('The silhouette plot for the various clusters.')
        ax1.set_xlabel('The silhouette coefficient values')
        ax1.set_ylabel('Cluster label')
        
        #vertical line for the mean SA score
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
        
        ax1.set_yticks([])
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        #plot to show the actual clusters
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(samples_arr[:, 0], samples_arr[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        
        #label the clusters
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c='white', alpha=1, s=200, edgecolor='k')
        
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        
        ax2.set_title('The visualization of the cluster data')
        ax2.set_xlabel('1st feature dim')
        ax2.set_ylabel('2nd feature dim')
        
        plt.suptitle(('Silhouette analysis for KMeans clustering on sample data with n_clusters = %d' % n_clusters),
                     fontsize=14, fontweight='bold')
    
    
    plt.show()
    return None


def set_bounds(crime_df):
    #manual LA bounds to remove egregious outliers due to faulty data (locations in Australia, Antarctica, etc.)
    bounding_box = geometry.Polygon([(-118.671136, 33.758599), (-118.71511, 34.350239), (-117.845436, 34.338900), (-117.916895, 33.696923)])
    la_city = {'city':['Los Angeles'],
               'geometry':bounding_box
               }
    la_city_df = pd.DataFrame(la_city, columns=['city', 'geometry'])
    
    la_bbox = gpd.GeoDataFrame(la_city_df, geometry=la_city_df['geometry'])
    la_bbox.crs = {'init':'epsg:4326'}
    
    
    crime_df = gpd.sjoin(crime_df, la_bbox, op='within', how='inner')
    return crime_df


def crime_date_indexing(crime_df):
    
    #using date as the index
    crime_df['date_occ'] = pd.to_datetime(crime_df['date_occ'])
    crime_df = crime_df.set_index(['date_occ'])
    crime_df.sort_index(ascending=True, inplace=True)
    
    #only columns we want
    crime_df = crime_df[['crm_cd_des', 'geometry']]
    
    #three years
    crime_df = crime_df.loc['2017-01-01':'2020-01-01']
    return crime_df

def filter_crimes(crime_df, crimes):
    crime_df = crime_df[crime_df['crm_cd_des'].isin(crimes)]
    return crime_df


os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
crime_df = gpd.read_file('fullcrime/full_crimes.shp')

#remove outliers based on city bounds
crime_df = set_bounds(crime_df)

#clean up index, filter to last 3 years
crime_df = crime_date_indexing(crime_df)

#get rid of facetious crime reports
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc/')
from crime_list import select_crimes
crime_df = filter_crimes(crime_df, select_crimes)



#make numpy array of the geometry
crime = np.array(list(crime_df.geometry.apply(lambda x: (x.x, x.y))))

        
# #set columns as axes
# plt.scatter(crime[:,0], crime[:,1])

# plt.show()

silhouette_analysis(crime)
        
        
        
''' 
For n_clusters= 15 The average silhouette score is:  0.4418387260547816
For n_clusters= 16 The average silhouette score is:  0.4381448701550302
For n_clusters= 17 The average silhouette score is:  0.4323762225017948
'''
        
        
        
        
        
        
        
        