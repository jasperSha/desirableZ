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

def ckdnearest(gdf1, gdf2, gdf2_cols=['gsId']):
    
    nA = np.array(list(zip(gdf1.geometry.x, gdf1.geometry.y)) )
    nB = np.array(list(zip(gdf2.geometry.x, gdf2.geometry.y)) )
    
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    
    gdf = pd.concat(
        [gdf1.reset_index(drop=True), gdf2.loc[idx, gdf2.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist, name='distance')], axis=1)
    

    return gdf


os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')

crime_df = gpd.read_file('fullcrime/full_crimes.shp')
schools_df = gpd.read_file('greatschools/joined.shp')
zillow_first_df = gpd.read_file('zillsecondrun/zillsecondrunfirsthalf.shp')
zillow_second_df = gpd.read_file('zillsecondrun/zillsecondrunsecondhalf.shp')
lausd_bounds_df = gpd.read_file('lausd/LOS_ANGELES_UNIFIED_SCHOOL.shp')


print(crime_df.head(), crime_df.shape)
print(schools_df.head(), schools_df.shape)
print(zillow_first_df.head(), zillow_first_df.shape)
print(zillow_second_df.head(), zillow_second_df.shape)
print(lausd_bounds_df.head(), lausd_bounds_df.shape)







