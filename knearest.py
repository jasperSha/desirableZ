import os
import pandas as pd
import numpy as np
import itertools
from operator import itemgetter
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree


#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = 10



os.chdir('/Users/Jasper/Documents/HousingMap/R_data/rental/')
properties = gpd.read_file('property.gpkg')
         

os.chdir('/Users/Jasper/Documents/HousingMap/R_data/schools/')
schools = gpd.read_file('school_package.gpkg')



def ckdnearest(gdf1, gdf2, gdf2_cols=['gsId']):
    
    nA = np.array(list(zip(gdf1.geometry.x, gdf1.geometry.y)) )
    nB = np.array(list(zip(gdf2.geometry.x, gdf2.geometry.y)) )
    
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    
    gdf = pd.concat(
        [gdf1.reset_index(drop=True), gdf2.loc[idx, gdf2.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist, name='distance')], axis=1)
    

    return gdf

print(ckdnearest(properties, schools))

# print(schools.head())
# schools = schools.reset_index(drop=True)
# print(schools.head())