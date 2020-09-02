import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import time


'''
task: to determine the local crime mesh values for a given house,
      aggregate then append to the house attributes vector.
     
        
Current method: use ball tree, 'haversine' metric, use n
'''

def geo_knearest(origins_df: gpd.GeoDataFrame, neighbors_df: gpd.GeoDataFrame, impute: bool=True, k: int=10) -> list:
    """
    

    Parameters
    ----------
    origins_df : gpd.GeoDataFrame
        houses dataframe with 'geometry' in WKT format.
    neighbors_df : gpd.GeoDataFrame
        schools in the same district as the properties dataframe with 'geometry' in WKT format.

    Returns
    -------
    List of lists of 2-tuples: (gsID, distance) of neighboring k=10 schools to each property in origins_df 
                               (in the same district). each list correponds to the same index of the individual
                               row in the origins_df (one to many relationship)
                               
                               if NOT impute: returns dictionary of lists for each housing property, ie
                               { 'zpid' : [(gsID, distance), (gsID, distance) ...] }

    """
    
    #scipy's cKDTree spatial index's query method
    #building a kd tree: time: O(nlogn)  space: O(kn)
    #knn search ~O(logn)
    
    #first reset indices so they line up
    origins_df.reset_index(drop=True, inplace=True)
    neighbors_df.reset_index(drop=True, inplace=True)
    
    #create numpy array out of the geometry of each dataframe
    origins = np.array(list(origins_df.geometry.apply(lambda x: (x.x, x.y))))
    neighbors = np.array(list(neighbors_df.geometry.apply(lambda x: (x.x, x.y))))
    
    #create the binary tree from which to query the neighbors
    btree = cKDTree(neighbors)
    
    #looking for 5 nearest for the average schools rating, but store k=3 for reference
    #finds distance, and index in second gdf of each neighbor
    dist, idx = btree.query(origins, k)
    
    #using the dataframe index to find the gsID of the neighboring schools
    gsids = []
    for school_idx in idx:
        gsids.append(neighbors_df['gsId'].iloc[school_idx])
    
    
    id_dist = []
    for tup in list(zip(gsids, dist)):
        id_dist.append(list(zip(tup[0],tup[1])))
          
    if impute:
        return id_dist
    else:
        #aggregating for property, explicitly returning the associated zpid
        zpids = origins_df['zpid'].tolist()
        zp_id_dist = dict(zip(zpids, id_dist))
        return zp_id_dist
    return
    

def vector_mask(v, dtype):
    '''
    converts v, a list of variable-length lists, to np array, padding with zeros
    for uniformity, dtype as specified.
    '''
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.zeros(mask.shape, dtype=dtype)
    out[mask] = np.concatenate(v)
    return out
    

def knearest_balltree(houses_df: gpd.GeoDataFrame, crime_mesh: gpd.GeoDataFrame, radius: float):
    '''
    Parameters
    ----------
    houses_df : gpd.GeoDataFrame
            current houses dataset, geometry in WKT format
    crime_mesh_df : gpd.GeoDataFrame
            crime density mesh with geometry in WKT format
    radius : float
            search radius for balltree in KILOMETERS
    k : int
        number of mesh points to aggregate for house rating
    
    Returns
    ---------
    houses_df with associated crime density of area around each house appended to end of df
    
    '''
    #first reset indices so they line up
    houses_df.reset_index(drop=True, inplace=True)
    crime_mesh.reset_index(drop=True, inplace=True)
    
    #create numpy array out of the geometry of each dataframe
    houses = np.array(list(houses_df.geometry.apply(lambda x: (x.x, x.y))))
    crimes = np.array(list(crime_mesh.geometry.apply(lambda x: (x.x, x.y))))
    
    #convert degrees to radians for balltree
    houses_rad, crimes_rad = np.radians(houses), np.radians(crimes)
    
    #convert radius kilometers to radians, using earth radius
    radius = radius / 6378.1
    
    #create ball tree of crime mesh, using haversine distances
    btree = BallTree(crimes_rad, metric='haversine')
    
    
    density_idx = btree.query_radius(houses_rad, r=radius)
    
    #list of arrays of densities for each house
    density_arr = [crime_mesh['density'].iloc[density_ind] for density_ind in density_idx]
    
    
    # arr = vector_mask(density_arr, dtype='float')
    
    
    #fill np array with densities, using zeroes to pad out variable lengths
    arr = np.zeros([len(density_arr), len(max(density_arr, key = lambda x: len(x)))])
    for i, j in enumerate (density_arr):
        arr[i][0:len(j)] = j
    
    '''
    vector masking time:  2.059340715408325
    enumeration time:  1.628880262374878
    
    '''
    #sum densities along row
    arr_sums = np.sum(arr, axis=1)
    zpids = np.array([houses_df['zpid']]).T
    arr_sums = np.column_stack([arr_sums, zpids])
    
    sums_df = pd.DataFrame(arr_sums, columns=['crime_density', 'zpid'])
    
    houses_df = houses_df.merge(sums_df, on='zpid', how='left')
    
    
    
    return houses_df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    