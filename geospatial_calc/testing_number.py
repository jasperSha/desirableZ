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
    



os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/testdatadensity')
test_data = pd.read_csv('test_data.csv')
test_data['geometry'] = test_data['geometry'].apply(loads)
data = gpd.GeoDataFrame(test_data, geometry=test_data['geometry'])
data.crs = {'init':'epsg:4326'}

coords = np.array(list(data.geometry.apply(lambda x: (x.x, x.y))))


earthradius = 6371 #kilometers
gridsize = 0.2 #kilometers

#radius will be derived from the kernel bandwidth
radius = 1.742 # 1.1km across


def kmToLat(km):
    #assuming close to equator
    return km/111.2


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



#define an event center (index still in order)
latc = lat[10]
lonc = lng[10]


'''
TODO: USE LINSPACE INSTEAD OF ARANGE TO ACCOUNT FOR FLOATING POINT FALLOFF
'''


#generate square mesh with 
lat_vec = np.arange(latc - radius, latc + radius, gridsize)
# print(lat_vec.shape)

lon_vec = np.arange(lonc - radius, lonc + radius, gridsize)
# print(lon_vec.shape)

print(latc + radius)
print(latc - radius)
print(gridsize)

print(lonc + radius)
print(lonc - radius)
print(gridsize)


#derive spherical pythagoras to find the tolerance vector magnitude
phi = np.cos(radius)
sigma = np.cos(lat_vec - latc)
lat_vec_t = np.arccos(phi/sigma)

#generate more normally distributed range outwards from lat/lon event center
lat_vec_t /= np.cos(np.radians(lat_vec))

lat_before_round = lat_vec_t
#snap tolerance vectors to meshgrid
lat_vec_t = np.round(lat_vec_t/gridsize, 0) * gridsize

#generate matrix with all lon coords in meshgrid
square = len(lon_vec)
lon_matrix = np.vstack([lon_vec]*square)

#subtract event center point to recenter on zero
diff_lon_matrix = abs(lon_matrix - lonc)


#using latitude tolerance vectors, zero out points not in neighborhood
# temp = lat_vec_t - diff_lon_matrix


