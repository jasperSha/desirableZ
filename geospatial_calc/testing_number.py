from numba import cuda
from numba import jit
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import pandas as pd
import geopandas as gpd
import os
from shapely.wkt import loads
import matplotlib.pyplot as plt
import math
import datetime
    



os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/testdatadensity')
test_data = pd.read_csv('test_data.csv')
test_data['geometry'] = test_data['geometry'].apply(loads)
data = gpd.GeoDataFrame(test_data, geometry=test_data['geometry'])
data.crs = {'init':'epsg:4326'}

data.to_file('test_data.gpkg', driver='GPKG')

coords = np.array(list(data.geometry.apply(lambda x: (x.x, x.y))))


earthradius = 6371 #kilometers
gridsize = 0.2 #kilometers

#radius will be derived from the kernel bandwidth
radius = 1.742 # 1.1km across


def kmToLat(km):
    #assuming close to equator
    return km/111.2

def date_diff(eventDate):
    if eventDate is None:
        return None
    today = datetime.date.today()
    date_obj = datetime.datetime.strptime(eventDate, '%Y-%m-%d').date()
    diff = today - date_obj
    return diff.days



#converting input numbers from kilometers to lat/lng degrees for greater accuracy
gridsize = round(kmToLat(gridsize), 3)

#specify input radius as kilometers
rad_km = radius

#convert radius km to radius in degrees
rad_degree = kmToLat(rad_km)

#number of grid steps
rad_steps = round(rad_degree/gridsize)

# #radius in km rounded off to nearest gridstep
# rad_km = rad_steps * gridsize * 111.2

#radius used for spherical pythagorean calc
radius = rad_steps * gridsize




#split lng/lat into separate vectors
lng = coords[:,0]
lat = coords[:,1]


#snap lat/lng event coords to adhere to gridsize then broadcast --> creates mesh only where the points exist
lng = lng / gridsize
lng = np.round(lng, 0)
lng *= gridsize
lng = lng[:,None]


lat = lat * (1/gridsize)
lat = np.round(lat, 0)
lat *= gridsize
lat = lat[:,None]

#generate date count for temporal trend analysis
data['dayCount'] = 0
data['dayCount'] = data['date_occ'].apply(date_diff)




'''
prior to here, all calculations are universal
'''



#define an event center (index still in order) here is where the iteration begins
latc = lat[10]
lonc = lng[10]


crime_center = test_data.iloc[10]


#generate square mesh with linspace, using range of 2n + 1
lat_vec = np.linspace(latc - radius, latc + radius, rad_steps * 2 + 1)

lon_vec = np.linspace(lonc - radius, lonc + radius, rad_steps * 2 + 1)


#derive spherical pythagoras to find the tolerance vector magnitude
phi = np.cos(radius)
sigma = np.cos(lat_vec - latc)
lat_vec_t = np.arccos(phi/sigma)

#generate more normally distributed range outwards from lat/lon event center
lat_vec_t /= np.cos(np.radians(lat_vec))

#snap tolerance vectors to meshgrid
lat_vec_t = np.round(lat_vec_t/gridsize, 0) * gridsize

#generate square mesh with all lon coords
square = len(lon_vec)
lon_matrix = np.vstack([np.transpose(lon_vec)]*square)

#subtract event center point to recenter on zero to compare to latitude tolerance vector magnitudes
lon_matrix_t = abs(lon_matrix - lonc)

#zero out points not within tolerance vector magnitude
temp = lat_vec_t - lon_matrix_t
temp[temp < (gridsize - (1e-6))] = 0
temp[temp > 0] = 1
lon_matrix_trunc = temp * lon_matrix


#matrix of all the latitudes 

#reverse lat elements first before transpose for square matrix, descending top to bottom
flipped_lat_vec = np.flipud(lat_vec)
lat_matrix = np.transpose(np.vstack([np.transpose(flipped_lat_vec)]*square))

#combine pairwise lat/lon matrices
lat_lon_matrix = np.array((lat_matrix, lon_matrix_trunc)).T

#remove all elements where lon was zeroed out
mask = lat_lon_matrix[:,:,1]
local_event_mesh = lat_lon_matrix[mask != 0]


#assign weights/dates to each point on the return meshgrid
weight, sumDate = np.tile(crime_center['weight'], (len(local_event_mesh),1)), np.tile(crime_center['dayCount'], (len(local_event_mesh),1))


return_mesh = np.column_stack((local_event_mesh,weight,sumDate))





# x = local_event_mesh[:,1]
# y = local_event_mesh[:,0]

# og_x = coords[:,0]
# og_y = coords[:,1]

# plt.scatter(x, y)
# plt.show()
# # plt.scatter(og_x,og_y)
# # plt.show()

# lon_max, lon_min = max(og_x), min(og_x) 
# lat_max, lat_min = max(og_y), min(og_y)













