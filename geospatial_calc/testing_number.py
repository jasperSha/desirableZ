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


    
def haversine(data):
    lat = data[:,0]
    lng = data[:,1]
    
    newlatdim = lat[:, None]
    diff_lat = newlatdim - lat
    diff_lng = lng[:, None] - lng
    
    
    #haversine
    d = np.sin(diff_lat/2)**2 + np.cos(lat[:,None])*np.cos(lat) * np.sin(diff_lng/2)**2
    dist = 2 * earthradius * np.arcsin(np.sqrt(d))
    print(dist)
    



os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/testdatadensity')
test_data = pd.read_csv('test_data.csv')
test_data['geometry'] = test_data['geometry'].apply(loads)
data = gpd.GeoDataFrame(test_data, geometry=test_data['geometry'])
data.crs = {'init':'epsg:4326'}

coords = np.array(list(data.geometry.apply(lambda x: (x.x, x.y))))


earthradius = 6371 #kilometers
gridsize = 0.1 #kilometers
radius = 1.1 #step count -> 1km across

def kmToLat(km):
    #assuming close to equator
    return km/111.2
def kmtoLng(km, lat):
    lati = np.radian(lat)
    dist_radian = km/111.320 * np.cos(lati) #conversion depends on latitude due to convergence/divergence from pole to pole
    return 


def calc_density(lati, loni, gridsize, radius, count, sumDate):
    #lati/loni are the lat/lng of the individual event
    _range = radius * gridsize
    lat_vec = np.linspace(lati - _range, lati + _range, gridsize)
    return lat_vec


#converting input numbers from kilometers to lat/lng degrees for greater accuracy
gridsize = round(kmToLat(gridsize), 3)

#specify input radius as kilometers
rad_km = radius

#convert radius km to radius in degrees
rad_degree = kmToLat(rad_km)

#whole number steps in grid to divide radius(in degrees) by gridsize(in degrees)
rad_steps = round(rad_degree/gridsize)

#radius in km
rad_km = rad_steps * gridsize * 111.2
radius = rad_steps

print(gridsize, rad_steps, rad_km, radius)


lng = coords[:,0]
lat = coords[:,1]

#round lat/lng to nearest gridpoints
lng = lng * (1/gridsize)
lng = np.round(lng, 0)
lng *= gridsize
print(lng)

lat = lat * (1/gridsize)
lat = np.round(lat, 0)
lat *= gridsize
print(lat)


