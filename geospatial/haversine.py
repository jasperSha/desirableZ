import numpy as np

def haversine(lon1, lat1, lon2, lat2, convert_rad=True):
    '''
    Parameters
    --------
    
    lon1, lat1 : nd.array of first points
    lon2, lat2 : nd.array of second points
    if convert_rad = True, convert lat/lon pairs to radian, default is True
    if looking for distance from a center out to multiple, just vstack the center points
    
    Returns
    radian_distance: 
        nd.array of haversine distance between first and second arrays, in km
    '''
    if convert_rad:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    
    km = 6367 * c
    return km
    
