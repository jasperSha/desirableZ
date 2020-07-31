import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import sys
from numba import njit, jit
from scipy.spatial import distance
from crime_density import full_crime_compile, fullcrime_kmeans
import time
from crime_list import misc_crime, property_crime, violent_crime, deviant_crime

#display entire numpy array
np.set_printoptions(threshold=sys.maxsize)


'''
Using the quartic kernel distribution for the density estimation.

The quartic kernel distribution function weighs near points more than far points, and
while the falloff is gradual, it is not as gradual as a normal distribution is;
thus the quartic function is determined to be more closely
aligned with the real life distribution of crime, e.g. high-crime areas that are
adjacent to more upscale neighborhoods with the dropoff being quite dramatic (at least
                                                                              more dramatic than what a
                                                                              normal distribution might
                                                                              follow.)
The quartic kernel is also less granular than a negative exponential or triangular
distribution, which have much more dramatic falloffs but produce finer variations;
due to computational/optimization concerns, the quartic also has the advantage of the
other two, as despite their greater accuracy, using those distributions for a dataset
of over a million points is more cumbersome.

Quartic function equation:
let dn = distance / h, where h is the chosen bandwidth
then,
P(x) = KWI*(15/16)*(1-dn**2)**2,
where the density value is comprised of K, a constant, W, a weight, and I, the intensity

The determination of bandwidth has the most impact on the resultant output however.

We will use locally adaptive bandwidth, adjusting to each cluster found through prior
silhouette analysis.

Using Silverman's Rule of Thumb bandwidth estimation formula, extrapolated to
two dimensions, we derive the formula for determining h as such:
    1. Calculate the mean center of input points (done a priori)
    2. Calculate distance from (weighted) mean center for all points (here is where weighted value of the points are factored in)
    3. Calculate interquartile range of distances (IQR)
    4. Calculate Standard Distance (simply standard deviation of the distances) as SD
    
    5**. Finally, h = 0.9 * min(SD, IQR/1.34) * n**-0.2,
       where Dm is the median distance.

**This formula for h optimizes the bandwidth by minimizing the 
mean integrated square error (MISE); however, this formula relies on the data
to follow a mostly normal distribution. in some cases, a bimodal distribution
using this bandwidth estimator can give wildly inaccurate results. However,
we will attempt to mitigate this possibility through accurate binning of the data.
(most likely going to be using longitude/latitude precision of ~0.01)



using ArcGIS's bandwidth variation on Silverman's Rule of Thumb,
h = 0.0010232459928684096 for 14 clusters
h = 0.0011021185631527905 for 15 clusters (slightly larger kernel)

using standard variation of Silverman's Rule of Thumb,
h = 0.0010232459928684096 for 14 clusters
h = 0.0011021185631527905 for 15 clusters

Seems identical, and runtimes are about the same for each.

'''

def kernelbandwidth(cluster_group: np.array, cluster_center: np.array) -> tuple:
    '''
    Parameters
    -------
    cluster_points: 2d array
                    array of geographical points.
    
    cluster_center: 2d array
                    geographical center of cluster
                    
    
    Returns
    -------
    h : float
        optimal bandwidth for kernel density estimation

    '''
    #tweak dimensions for cdist
    cluster_group_arr = np.vstack(cluster_group)
    center = np.array([cluster_center])
    
    dist_array = distance.cdist(center, cluster_group_arr, 'euclidean')
    standard_distance = np.std(dist_array)
    N = cluster_group_arr.size
    
    q1 = np.percentile(cluster_group_arr, 25, interpolation='midpoint')
    q3 = np.percentile(cluster_group_arr, 75, interpolation='midpoint')
    IQR = q3 - q1
    
    h = 0.9 * min(standard_distance, IQR/1.34) * N**-0.2
    
    return h



def quartic(d,h):
    #for more concentrated peaks, steep falloff
    u=d/h
    
    P=(15/16)*(1-u**2)**2
    return P

def gaussian(d, h):
    #for normal distribution
    u = d/h
    
    k = 1/math.sqrt(2*math.pi)
    P = k * math.exp(-(u**2)/2)
    return P

def epanechnikov(d, h):
    #rounded out peaks, gradual falloff
    u = d/h
    
    P = (3/4)*(1 - u**2)**2
    return P
    
#PROCESSING
# @jit(nopython=True)
def generate_intensity(x, y, xc, yc):
    intensity_list=[]
    for j in range(len(xc)):
        intensity_row=[]
        for k in range(len(xc[0])):
            kde_value_list=[]
            for i in range(len(x)):
                #CALCULATE DISTANCE
                d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
                if d<=h:
                    p=gaussian(d,h)
                else:
                    p=0
                kde_value_list.append(p)
            #SUM ALL INTENSITY VALUE
            p_total=sum(kde_value_list)
            intensity_row.append(p_total)
        intensity_list.append(intensity_row)
    return intensity_list




crime_df, crime_coords = full_crime_compile()
clusters_df, clusters, centers = fullcrime_kmeans(crime_df, crime_coords, n_clusters=15)

#first cluster group
firstcluster = np.array(list(clusters[1].geometry.apply(lambda x: (x.x, x.y))))
print(firstcluster[10:50])
h = kernelbandwidth(firstcluster, centers[1])

x, y = np.split(firstcluster, 2, 1)


#DEFINE GRID SIZE AND RADIUS(h)
#grid_size 3rd decimal place for the area of a large field
grid_size=0.005
#h is our kernel radius to determine cluster influence


#GETTING X,Y MIN AND MAX
x_min=min(x)
x_max=max(x)
y_min=min(y)
y_max=max(y)

#CONSTRUCT GRID
x_grid=np.arange(x_min-h,x_max+h,grid_size)
y_grid=np.arange(y_min-h,y_max+h,grid_size)
x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)

#GRID CENTER POINT

xc=x_mesh+(grid_size/2)

yc=y_mesh+(grid_size/2)


#HEATMAP OUTPUT
# for row in intensity_list:
#     print(row)
intensity_list = generate_intensity(x, y, xc, yc)
intensity=np.array(intensity_list)
print(intensity)
plt.pcolormesh(x_mesh,y_mesh,intensity)
# plt.scatter(x,y, s=0.5, alpha=0.5)
# plt.colorbar()
plt.show()



'''
following is for checking cluster center accuracy
'''
# x, y = [], []
# for arr in cluster_group:
#     x.append(arr[0])
#     y.append(arr[1])
# x2 = centers[0][0]
# y2 = centers[0][1]

# colors = (0, 0, 0)
# plt.scatter(x, y, s=1, c=colors, alpha=0.5)
# plt.scatter(x2, y2, s=2, c='blue', alpha=1)
# plt.show()



