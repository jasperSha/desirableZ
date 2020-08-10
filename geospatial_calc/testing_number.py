import numpy as np
import pandas as pd
import os
from shapely.wkt import loads
import matplotlib.pyplot as plt
import math
import datetime, time
import seaborn as sns


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


def generate_event_mesh(latc, lonc, weight, sumDate, count):
            
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
    
    #remove all elements where lon was zeroed out by tolerance vector
    mask = lat_lon_matrix[:,:,1]
    local_event_mesh = lat_lon_matrix[mask != 0]
    
    
    #assign weights/dates to each point on the return meshgrid
    weight, sumDate, count = np.tile(weight, (len(local_event_mesh),1)), np.tile(sumDate, (len(local_event_mesh),1)), np.tile(count, (len(local_event_mesh),1))
    
    
    return_mesh = np.column_stack((local_event_mesh,weight,sumDate, count))
    return return_mesh

def agg_func(x):
    d = {}
    count = x['count'].sum()
    daycount = x['sumDate'].sum()
    weight = x['weight'].mean()
    d['date_avg'] = daycount / count
    d['density'] = weight * count
    d['lng'] = x['lng'].max()
    d['lat'] = x['lat'].max()
    return pd.Series(d)

def agg_arr(x):
    density = np.average(x[:,4], weights=x[:,2])
    dateavg = np.average(x[:,3], weights=x[:,4])
    return x[:,0].max(), x[:,1].max(), density, dateavg




os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/testdatadensity')
data = pd.read_csv('test_data.csv')


data['geometry'] = data['geometry'].apply(loads)
coords = np.array(list(data.geometry.apply(lambda x: (x.x, x.y))))


#generate date count for temporal trend analysis
data['dayCount'] = 0
data['dayCount'] = data['date_occ'].apply(date_diff)

gridsize = 0.2 #kilometers

#radius will be derived from the kernel bandwidth
radius = 1.1 # 1.742km across


#converting input numbers from kilometers to lat/lng degrees for greater accuracy
gridsize = round(kmToLat(gridsize), 3)

#specify input radius as kilometers
rad_km = radius

#convert radius km to radius in degrees
rad_degree = kmToLat(rad_km)

#number of grid steps
rad_steps = round(rad_degree/gridsize)

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

snapped_latlng = np.concatenate((lat, lng), axis=1)

weight = data['weight']
dayCount = data['dayCount']

#combine, convert to df for aggregate functions
snapped_latlng = np.column_stack((snapped_latlng, weight, dayCount))
df = pd.DataFrame(snapped_latlng, columns=['lat', 'lng', 'weight', 'dayCount'])
df = df.sort_values(['lat', 'lng'], ascending=True).reset_index(drop=True)
df = df.groupby(['lat', 'lng']).agg(weight=('weight', 'mean'),sumDate=('dayCount', 'sum'),count=('lng', 'count')).reset_index()

inventory_array = df.to_numpy()

#generate full gridmesh
full_grid = np.concatenate([generate_event_mesh(latc=event[0], lonc=event[1], weight=event[2], sumDate=event[3], count=event[4]) for event in inventory_array])

#sort grid for np grouping
sorted_grid = full_grid[np.lexsort((full_grid[:,0], full_grid[:,1]))]

#pull out lat/lng 2d array to sort and find unique indexes
lat, lng = sorted_grid[:,[0]], sorted_grid[:,[1]]
sorted_latlng = np.hstack((lat, lng))
unique, idx, counts = np.unique(sorted_latlng, axis=0, return_index=True, return_counts=True)


#split, group by unique
split_grid = np.split(sorted_grid, np.sort(idx))

#get rid of any empty elements
split_grid = [x for x in split_grid if x.size != 0]

#consolidate and normalize density
final_grid = np.array([agg_arr(gridpoint) for gridpoint in split_grid])
final_grid[:,2] = final_grid[:,2] / final_grid[:,2].max()


density = final_grid[:,2]
_, idx, counts = np.unique(density, return_index=True, return_counts=True)
density = np.column_stack((density[idx], counts))



#plotting
cmap = sns.cubehelix_palette(as_cmap=True)
x, y = final_grid[:,1], final_grid[:,0]
f, ax = plt.subplots()
points = ax.scatter(x,y, s=0.001, cmap=cmap)
f.colorbar(points)










