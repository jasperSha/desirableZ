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
def silhouette_analysis(samples_arr: np.array, cluster_range: list) -> list:
    '''
    

    Parameters
    ----------
    samples_arr : TYPE
        numpy array of geographical points.
    cluster_range : 
        range of potential numbers of clusters.

    Returns
    -------
    list of n_clusters and their silhouette score; the closer the score to 1, the more optimal.

    '''
    #possible n clusters derived from ArcGIS heatmap peaks for LA county only
    
    
    for n_clusters in cluster_range:
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
    crime_df = crime_df.loc['2015-01-01':'2020-01-01']
    return crime_df

def filter_crimes(crime_df, crimes):
    crime_df = crime_df[crime_df['crm_cd_des'].isin(crimes)]
    return crime_df

def build_coords():
    os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
    crime_df = gpd.read_file('fullcrime/full_crimes.shp')
    
    #remove outliers based on city bounds
    crime_df = set_bounds(crime_df)
    
    #clean up index, filter to last 3 years
    crime_df = crime_date_indexing(crime_df)
    
    #filter crime by severity
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc/')
    from crime_list import misc_crime, violent_crime, property_crime, deviant_crime
    
    misc_df = filter_crimes(crime_df, misc_crime)
    violent_df = filter_crimes(crime_df, violent_crime)
    prop_df = filter_crimes(crime_df, property_crime)
    deviant_df = filter_crimes(crime_df, deviant_crime)
    
    
    #make numpy array of the geometry
    misc_coords = np.array(list(misc_df.geometry.apply(lambda x: (x.x, x.y))))
    viol_coords = np.array(list(violent_df.geometry.apply(lambda x: (x.x, x.y))))
    prop_coords = np.array(list(prop_df.geometry.apply(lambda x: (x.x, x.y))))
    devi_coords = np.array(list(deviant_df.geometry.apply(lambda x: (x.x, x.y))))
    
    
    crime_coords = [
            viol_coords,
            devi_coords,
            prop_coords,
            misc_coords
        ]
    
    crime_categories = [
        'violent crime',
        'deviant crime',
        'property crime',
        'misc crime'
        ]
    return crime_coords, crime_categories

# cluster_range = [4, 5, 6, 7, 8, 9, 10]
# cl_range = [9, 10]

# print('Silhouette analysis for miscellaneous crime')
# silhouette_analysis(misc_coords, cluster_range)


# for crime, cat in list(zip(crime_coords, crime_categories)):
#     print('Silhouette analysis for %s'%cat)
#     silhouette_analysis(crime, cluster_range)

# #optimal k clusters found for each category (in order of crime_categories) 2017 to 2020
# k_clusters = [7, 7, 5, 6]

#optimal k clusters found for each category (in order of crime_categories) 2015 to 2020
# k_clusters = [5, 7, 5, 6]


def cluster_centers(samples_arr: np.array, crime_categories: list, crime_coords: list, k_clusters: list, n_clusters: int):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer.fit_predict(samples_arr)
    centers = clusterer.cluster_centers_
    
    print('\n', centers)

    count = 0
    for category, clusters in list(zip(crime_coords, k_clusters)):
        print('Finding the cluster centers of %s' %crime_categories[count])
        cluster_centers(category, clusters)
        count += 1


#quartic as crime patterns theoretically follow a normal distribution, though this topic has not been fully addressed
def kde_quartic(d,h):
    dn=d/h
    P=(15/16)*(1-dn**2)**2
    return P



    
'''
#GPS precision for parcel of land determines grid_size
grid_size = 0.001 (3rd decimal place for the area of a large agricultural field)

h= 0.01 (2nd for influence of village to another)
'''
    
    
    
    
    
    
    






    
    
    
    
        
''' 

Rankings:
    violent
    deviant
    property
    misc


FOR THE PERIOD OF 2015 TO 2020:

n_clusters for:
violent: 5
deviant: 7
property: 5
misc: 6

Finding the cluster centers of violent crime

 [[-118.27517812   34.02880825]
 [-118.42361836   34.21596466]
 [-118.28303636   33.78022415]
 [-118.56469308   34.2101801 ]
 [-118.42645033   34.02420681]]
Finding the cluster centers of deviant crime

 [[-118.30466966   34.05184964]
 [-118.56381299   34.2118    ]
 [-118.42976949   34.22602904]
 [-118.28526512   33.77426865]
 [-118.1855064    34.0750253 ]
 [-118.27809354   33.97028884]
 [-118.45440441   34.01521882]]
Finding the cluster centers of property crime

 [[-118.27742404   34.0397772 ]
 [-118.56285214   34.21070311]
 [-118.43049736   34.02517694]
 [-118.41912416   34.2053919 ]
 [-118.28554002   33.78655923]]
Finding the cluster centers of misc crime

 [[-118.48534383   34.01836123]
 [-118.28541397   34.03160509]
 [-118.08059044   34.09209838]
 [-118.42885777   34.21712034]
 [-118.56977967   34.20616979]
 [-118.28194502   33.77780458]]


    
Silhouette analysis for violent crime
For n_clusters= 4 The average silhouette score is:  0.4866150894127862
For n_clusters= 5 The average silhouette score is:  0.49264815868135864 #
For n_clusters= 6 The average silhouette score is:  0.4459419214026837
For n_clusters= 7 The average silhouette score is:  0.4611975741352134
For n_clusters= 8 The average silhouette score is:  0.44465717467602484
For n_clusters= 9 The average silhouette score is:  0.4259914535926751
For n_clusters= 10 The average silhouette score is:  0.43073852010999686


 Silhouette analysis for deviant crime
For n_clusters= 4 The average silhouette score is:  0.45959064281660306
For n_clusters= 5 The average silhouette score is:  0.45786695383870557
For n_clusters= 6 The average silhouette score is:  0.44852107730804974
For n_clusters= 7 The average silhouette score is:  0.4737495706006081 #
For n_clusters= 8 The average silhouette score is:  0.44925485321058944
For n_clusters= 9 The average silhouette score is:  0.4536815486703077
For n_clusters= 10 The average silhouette score is:  0.445255149904319

Silhouette analysis for property crime
For n_clusters= 4 The average silhouette score is:  0.46720342024132044
For n_clusters= 5 The average silhouette score is:  0.4801486284830978 #
For n_clusters= 6 The average silhouette score is:  0.42394171702537264
For n_clusters= 7 The average silhouette score is:  0.4382794791730499
For n_clusters= 8 The average silhouette score is:  0.42604062766795986    

Silhouette analysis for miscellaneous crime
For n_clusters= 4 The average silhouette score is:  0.7749747878637137
For n_clusters= 5 The average silhouette score is:  0.7890326893232749
For n_clusters= 6 The average silhouette score is:  0.790681078559728 #
For n_clusters= 7 The average silhouette score is:  0.7531691016395206
For n_clusters= 8 The average silhouette score is:  0.7570285730233564
For n_clusters= 9 The average silhouette score is:  0.48225722619853334
For n_clusters= 10 The average silhouette score is:  0.4827669510168




FOR THE PERIOD OF 2017 TO 2020
n_clusters for:
    violent: 7
    deviant: 7
    property: 5
    misc: 6

Finding the cluster centers of violent crime

 [[-118.45151412   34.0124254 ]
 [-118.32413231   34.06526875]
 [-118.28262862   33.77410265]
 [-118.42396278   34.21685481]
 [-118.28247014   33.97395464]
 [-118.23632137   34.05834918]
 [-118.56446515   34.2099804 ]]
 
Finding the cluster centers of deviant crime

 [[-118.43043895   34.22563193]
 [-118.27854408   33.96803486]
 [-118.18524682   34.07363618]
 [-118.28512088   33.77325522]
 [-118.56454426   34.21280439]
 [-118.45131969   34.01784329]
 [-118.302618     34.05134275]]
 
Finding the cluster centers of property crime

 [[-118.43222082   34.02460634]
 [-118.27995643   34.05277818]
 [-118.56361479   34.2101943 ]
 [-118.27699563   33.88880652]
 [-118.41965599   34.20491349]]
 
Finding the cluster centers of misc crime

 [[-118.48547445   34.01847941]
 [-118.08054463   34.09186934]
 [-118.28539372   34.03270877]
 [-118.42879105   34.2167742 ]
 [-118.28084153   33.77999705]
 [-118.57074682   34.20727909]]


Silhouette analysis for miscellaneous crimes
For n_clusters= 4 The average silhouette score is:  0.7803450217369586
For n_clusters= 5 The average silhouette score is:  0.7925708278405666
For n_clusters= 6 The average silhouette score is:  0.7945949742817022
For n_clusters= 7 The average silhouette score is:  0.7569681504572621
For n_clusters= 8 The average silhouette score is:  0.761738245621019
For n_clusters= 9 The average silhouette score is:  0.4876120348887703
For n_clusters= 10 The average silhouette score is:  0.4858918744563427
For n_clusters= 11 The average silhouette score is:  0.4850010461185952
For n_clusters= 12 The average silhouette score is:  0.4657703581202816
For n_clusters= 13 The average silhouette score is:  0.46545949808405057
For n_clusters= 14 The average silhouette score is:  0.45747888319550845

 Silhouette analysis for violent crimes
For n_clusters= 4 The average silhouette score is:  0.45408128179358803
For n_clusters= 5 The average silhouette score is:  0.44232605377567513
For n_clusters= 6 The average silhouette score is:  0.4466617091766663
For n_clusters= 7 The average silhouette score is:  0.4634825380143094
For n_clusters= 8 The average silhouette score is:  0.4477940405182154
For n_clusters= 9 The average silhouette score is:  0.42747112234010776
For n_clusters= 10 The average silhouette score is:  0.4323360278132864
For n_clusters= 11 The average silhouette score is:  0.4376877046494286
For n_clusters= 12 The average silhouette score is:  0.4190021459787022
For n_clusters= 13 The average silhouette score is:  0.4161816040334206
For n_clusters= 14 The average silhouette score is:  0.4088476019770265

Silhouette analysis for property crimes
For n_clusters= 4 The average silhouette score is:  0.44510826507033174
For n_clusters= 5 The average silhouette score is:  0.4613739930846025
For n_clusters= 6 The average silhouette score is:  0.4062359533362962
For n_clusters= 7 The average silhouette score is:  0.4394407973235615
For n_clusters= 8 The average silhouette score is:  0.42784712097806243
For n_clusters= 9 The average silhouette score is:  0.4160342740260717
For n_clusters= 10 The average silhouette score is:  0.4184097987196484
For n_clusters= 11 The average silhouette score is:  0.4138935228642149
For n_clusters= 12 The average silhouette score is:  0.41031383251966747
For n_clusters= 13 The average silhouette score is:  0.4178503048084896
For n_clusters= 14 The average silhouette score is:  0.42414359774287863

Silhouette analysis for deviant crimes
For n_clusters= 4 The average silhouette score is:  0.46050230317224206
For n_clusters= 5 The average silhouette score is:  0.4682914039827742
For n_clusters= 6 The average silhouette score is:  0.4505054784412173
For n_clusters= 7 The average silhouette score is:  0.4724689771122283
For n_clusters= 8 The average silhouette score is:  0.44895609167869477
For n_clusters= 9 The average silhouette score is:  0.45076801402856465
For n_clusters= 10 The average silhouette score is:  0.44390575640618546
For n_clusters= 11 The average silhouette score is:  0.4450192688107003
For n_clusters= 12 The average silhouette score is:  0.4274026946228304
For n_clusters= 13 The average silhouette score is:  0.41139991714713475
For n_clusters= 14 The average silhouette score is:  0.40827149632348647

'''
        
        
        
        
        
        
        
        