import numpy as np
import pandas as pd
import datetime


'''
Geospatial Point Density Algorithm, translated from R to Python.
As an inversion of the typical kernel density algorithm used to 
determine the population density of a region, the point density
algorithm seeks to bypass the typical problems inherent to the
kernel density algorithm.

The biggest problem with the kernel density algorithm is threefold:
   
The first and most impactful one is the choice of kernel bandwidth.
The bandwidth selection for smoothing out the kernels, depending on the
granularity of the data point distribution, can easily obscure important
points, or, if the bandwidth selected is too fine, then the runtime can
reach extreme levels, especially in the case here where the crime data
contains millions of points.

The other two parameters to consider are the distribution function and,
in the case of geospatial kernel density, the 2-dimensional bin sizes. Whether
using the quartic or normal distributions, depending on the location
of the grid, the aggregation of points in each bin can vary widely. 

For example, using the quartic distribution, which favors points located
near the center with a rapid fall-off approaching the edges, even adjusting the
meshgrid location by ~.001 latitudinal/longitudinal degrees, which corresponds
to about ~110 kilometers or about the size of a large agricultural field,
the resulting apparent density of that bin can significantly change, even
though it would be easy to argue that the crime density of a neighborhood
does NOT drastically change across the width of a field, but due to the 
artificial restraint of the meshsize and bandwidth search radius, it is
altered.


With all of these problems, I decided to use an inversion of the kernel
density algorithm. Instead of designating an arbitrary square mesh grid, 
overlaying it over an area that has naturally irregular distributions across 
geographic lines, a mesh would arise from the points themselves. 


First, we still have to generate a simulated mesh. Using the Silverman's Rule
of Thumb and the haversine formula for determining distance between two points
on a sphere (as explained in detail in the kerneldensity.py module), we find
a suitable search radius, about ~.0x degrees, or ~1.1 kilometers.

Once each point has been rounded to a certain radius stepsize, we then iterate
across the points. (In the future I do plan on revisiting this part, and 
                    dividing the crime data into blocks, then threads, and
                    perhaps work on parallelizing the density mesh generation)

For each point, we use our pre-determined gridsize to create small grid steps
along each latitude and longitude. Now, we will increase the value of the mesh
grid by the population density of that particular point (as the data points were
                                                         rounded off, some converge
                                                         and are aggregated onto
                                                         points).
Because we're looking for a circular generative mesh, we will use all of the
longitude coordinate points intersecting the center point, but limit the latitude 
coordinates.

The theorem states, where u, v, and w are vectors extending from the center of
the sphere to the surface, and a = uv, b = uw, and c = vw,

then: cos(c) = cos(a) * cos(b),

where a, b, and c are essentially the difference in radians from the angle
between uv, uw, and vw, AND, it is a special case where C is a right angle.

Thus we take the cosine of the desired search radius, the cosine of the
difference in latitude between each grid bar and the data point in the center,
and the arccos of their quotient is the length of the desired latitude vector.

Once we have the latitude vector corresponding to each longitude, we then zero
out all the points not included, and all that's left is to then
add the density of the center point to the value of the remaining mesh points.

Finally, we stack all of these numpy arrays with coordinates and density values,
stack according to common coordinate points, and sum up the remaining values.

Normalization and analysis of the resultant date values is left for another module.

'''

def kmToLat(km):
    #assuming close to equator
    return km/111.2

def latToKm(degrees):
    return degrees * 111.2

def date_diff(eventDate):
    if eventDate is None:
        return None
    today = datetime.date.today()
    date_obj = datetime.datetime.strptime(eventDate, '%Y-%m-%d').date()
    diff = today - date_obj
    return diff.days


def agg_arr(x):
    density = np.average(x[:,4], weights=x[:,2])
    dateavg = np.average(x[:,3], weights=x[:,4])
    return x[:,0].max(), x[:,1].max(), density, dateavg


def generate_event_mesh(latc, lonc, weight, sumDate, count, radius, rad_steps, gridsize):
    
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





def point_density(events: pd.DataFrame, coords: np.array, radius: float, gridsize: float) -> np.array:
    '''
    
    Parameters
    ----------
    events : pd.DataFrame
        crime data with coordinates in WKT format, associated weights and dates.
    radius : float
        bandwidth in kilometers derived from Silverman's Rule of Thumb for kernel density.
    gridsize : float
        granularity of the returned coordinate mesh, can be adjusted according to desired performance

    Returns
    -------
    Numpy array of coordinate mesh with point densities

    '''
    
    # events['geometry'] = events['geometry'].apply(loads)
    # coords = np.array(list(events.geometry.apply(lambda x: (x.x, x.y))))
    
    
    #generate date count for temporal trend analysis
    events['dayCount'] = 0
    events['dayCount'] = events['date_occ'].apply(date_diff)
    
    
    #converting input numbers from kilometers to lat/lng degrees for greater accuracy
    gridsize = round(kmToLat(gridsize), 3)
    
    #specify input radius as kilometers
    rad_km = radius
    
    #convert radius km to radius in degrees
    rad_degree = kmToLat(rad_km)
    
    #number of grid steps need explicit type conversion since Python3
    rad_steps = int(round(rad_degree/gridsize))
    
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
    
    weight = events['weight']
    dayCount = events['dayCount']
    
    #combine, convert to df for aggregate functions
    snapped_latlng = np.column_stack((snapped_latlng, weight, dayCount))
    df = pd.DataFrame(snapped_latlng, columns=['lat', 'lng', 'weight', 'dayCount'])
    df = df.sort_values(['lat', 'lng'], ascending=True).reset_index(drop=True)
    df = df.groupby(['lat', 'lng']).agg(weight=('weight', 'mean'),sumDate=('dayCount', 'sum'),count=('lng', 'count')).reset_index()
    
    inventory_array = df.to_numpy()
    
    #generate full gridmesh
    full_grid = np.concatenate([generate_event_mesh(latc=event[0], lonc=event[1], weight=event[2], sumDate=event[3], count=event[4], radius=radius, rad_steps=rad_steps, gridsize=gridsize) for event in inventory_array])
    
    #sort grid for np grouping (lexsort sorts b, then a)
    sorted_grid = full_grid[np.lexsort((full_grid[:,0], full_grid[:,1]))]
    
    #pull out lat/lng 2d array to sort and find unique indexes
    lat, lng = sorted_grid[:,[0]], sorted_grid[:,[1]]
    sorted_latlng = np.hstack((lat, lng))
    unique, idx, counts = np.unique(sorted_latlng, axis=0, return_index=True, return_counts=True)
    
    
    #split, group by unique
    split_grid = np.split(sorted_grid, np.sort(idx))
    
    #get rid of any empty elements
    split_grid = [x for x in split_grid if x.size != 0]
    
    #aggregate density and average date
    final_grid = np.array([agg_arr(gridpoint) for gridpoint in split_grid])
    
    
 
    return final_grid



if __name__ == '__main__':
    pass






