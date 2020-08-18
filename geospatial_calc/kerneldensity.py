import numpy as np
import math
import numpy as np
import sys
from numba import njit, jit
from scipy.spatial import distance
from silhouette import full_crime_compile, fullcrime_kmeans
import time
import os
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
    h : measurement in DEGREES
        optimal bandwidth for kernel density estimation (or point density radius)

    '''
    
    #tweak dimensions for cdist
    cluster_group_arr = np.vstack(cluster_group)
    center = np.array([cluster_center])
    
    
    dist_array = haversine_center(cluster_group_arr, center)
    standard_distance = np.std(dist_array)
    N = cluster_group_arr.size
    
    q1 = np.percentile(cluster_group_arr, 25, interpolation='midpoint')
    q3 = np.percentile(cluster_group_arr, 75, interpolation='midpoint')
    IQR = q3 - q1
    
    # h = 0.9 * min(standard_distance, IQR/1.34) * N**-0.2
    
    #R uses this formula for bandwidth
    h = 4 * 1.06 * min(standard_distance, IQR/1.34) * N**-0.2
    
    
    return h


'''
Two methods for kernel density estimation, varying on computational cost and accuracy.

One method: create hexagonal bins over the data points, sum data within each hexagon
then color code based off sum of each hexagon.
    pros:
        - computationally more efficient
        - hexagons can then be output as polygons for later crime mapping of zillow
          properties
    cons:
        - bin placement can drastically affect the density rating of the hexagons,
          as they dont arise naturally from the data, but rather are fitted based on
          prior parameters
          
Second method: 
    

'''



def quartic(d,h):
    #for more concentrated peaks, steep falloff
    u=d/h
    
    P=(15/16)*(1-u**2)**2
    return P

def gaussian(d, h):
    #for normal distribution
    u = d/h
    
    k = 1/math.sqrt(2*math.pi)
    P = k * math.exp(-1/2*(u**2))
    return P

def epanechnikov(d, h):
    #rounded out peaks, gradual falloff
    u = d/h
    
    P = (3/4)*(1 - u**2)**2
    return P
    

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.array([sum_x/length, sum_y/length])



