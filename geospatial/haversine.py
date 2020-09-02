import numpy as np

def haversine(lat1, lat2, lon1, lon2, earthradius=):
    '''
    Parameters
    --------
    cluster_points 2d array
    
    center 2d array (one element)
    
    Returns
    radian_distance: 
        np.array with each haversine distance in radians between each cluster 
        point and the center point
    '''
    
    cluster_points, center = np.radians(cluster_points), np.radians(center)
    
    #extend center array to match row-wise all cluster_points
    center = np.tile(center, (len(cluster_points), 1))
    
    lat1, lon1 = cluster_points[:,1], cluster_points[:,0]
    lat2, lon2 = center[:,1], center[:,0]
    
    #broadcast
    lat1, lon1 = lat1[:,None], lon1[:,None]
    lat2, lon2 = lat2[:,None], lon2[:,None]
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    
    radian_distance = c * 180/math.pi
    return radian_distance
    
