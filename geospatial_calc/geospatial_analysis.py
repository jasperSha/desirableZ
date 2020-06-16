import os
import pandas as pd
import numpy as np
import itertools
from operator import itemgetter
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn import preprocessing


#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = 10


def property_school_districts(zillow_df: gpd.GeoDataFrame, districts_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    zillow_df : GeoDataFrame
        Dataframe of housing properties
        
    districts_df : GeoDataFrame
        Dataframe of school districts 
        

    Returns
    -------
    sjoind_df : zillow_df with corresponding school districts
        

    """
      
    sjoined_df = gpd.sjoin(zillow_df, districts_df, op='within')
    
    #preserve property columns, just concat the school districts column-wise
    columns = zillow_df.columns.values.tolist()
    columns.append('DISTRICT')
    
    sjoined_df = sjoined_df[columns]
    
    return sjoined_df



def normalize_columns(zillow_df: pd.DataFrame) -> pd.DataFrame:
    
    
    arr = zillow_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(arr)
    return pd.DataFrame(x_scaled, columns=zillow_df.columns)


def ckdnearest_schools(zillow_df: gpd.GeoDataFrame, schools_df: gpd.GeoDataFrame, kn=3, kavg=5) -> gpd.GeoDataFrame:
    """
    
    Parameters
    ----------
    zillow_df : GeoDataFrame
        Dataframe of housing properties
        
    schools_df : GeoDataFrame
        Dataframe of schools
        
    kn : Integer
        Number of nearest schools for display
        
    kavg : Integer
        Number of nearest schools to calculate average GreatSchools rating

    Returns
    -------
    gdf : GeoDataFrame
        Housing Dataframe with average of the ratings and names of the kn schools.

    """
    
    
    
    #first we need to demarcate the properties and schools frame by district, then perform the knearest between each corresponding relation
    
    
    #first find nearest neighbors
    #scipy's cKDTree spatial index allows for a vectorized search instead of iterative
    
    
    nA = np.array()
    
    nA = np.array(list(zip(gdf1.geometry.x, gdf1.geometry.y)) )
    nB = np.array(list(zip(gdf2.geometry.x, gdf2.geometry.y)) )
    
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=3)
    
    gdf = pd.concat(
        [gdf1.reset_index(drop=True), gdf2.loc[idx, gdf2.columns != 'geometry'].reset_index(drop=True),
          pd.Series(dist, name='distance')], axis=1)
    

    return gdf

def split_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    

    Parameters
    ----------
    df : GeoDataFrame
        A GeoDataFrame of property listings

    Returns
    -------
    A GeoDataFrame .

    """
    return gdf




os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
zillow_df = gpd.read_file('zillow/zillow.shp')

# crime_df = gpd.read_file('fullcrime/full_crimes.shp')
schools_df = gpd.read_file('greatschools/joined.shp')

# zillow_knearest = ckdnearest(zillow_df, schools_df)
# print(zillow_knearest.head())

#testing knearest schools (using k=3)



school_districts_df = gpd.read_file('school_districts_bounds/School_District_Boundaries.shp')


sjoined_df = property_school_districts(zillow_df, school_districts_df)

print(sjoined_df.head())


# # LA_zillow_df = point_in_polygon(zillow_df, lausd_bounds_df)

# # LA_schools_df = point_in_polygon(schools_df, lausd_bounds_df)




# print(LA_zillow_df.shape)
# print(LA_schools_df.shape)
# print(LA_crime_df.shape)


# input_columns = ['low', 
#                  'high', 
#                  'zindexValu', 
#                  'lastSoldPr', 
#                  'bathrooms', 
#                  'bedrooms',
#                  'finishedSq',
#                  'lotSizeFt',
#                  'taxAssessm']

# output_columns = ['zestimate',
#                   'rentamount']

# zillow_input = zillow_df[input_columns]

# zillow_output = zillow_df[output_columns]


# # print(zillow_df.head(), zillow_df.dtypes)
# print(zillow_input.describe(), '\n', zillow_output.describe())





# zill_norm = normalize_columns(zillow_df)

# print(zill_norm.head())












