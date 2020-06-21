import os
import pandas as pd
import numpy as np
import itertools
from operator import itemgetter
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.impute import KNNImputer
from sklearn import preprocessing


#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = 10


def assign_property_school_districts(properties_df: gpd.GeoDataFrame, districts_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    zillow_df : GeoDataFrame
        Dataframe of housing properties, 'geometry' in WKT format
        
    districts_df : GeoDataFrame
        Dataframe of school district boundaries, 'geometry' in WKT format
        

    Returns
    -------
    sjoind_df : properties_df, each with their respective school district
        

    """
      
    sjoined_df = gpd.sjoin(properties_df, districts_df, op='within')
    
    #preserve property columns, just concat the school districts column-wise
    columns = properties_df.columns.values.tolist()
    columns.append('DISTRICT')
    
    sjoined_df = sjoined_df[columns]
    
    return sjoined_df



def normalize_columns(zillow_df: pd.DataFrame) -> pd.DataFrame:
    
    
    arr = zillow_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(arr)
    return pd.DataFrame(x_scaled, columns=zillow_df.columns)



def geo_knearest(origins_df: gpd.GeoDataFrame, neighbors_df: gpd.GeoDataFrame, k=10) -> tuple:
    """
    

    Parameters
    ----------
    origins_df : gpd.GeoDataFrame
        properties dataframe with 'geometry' in WKT format.
    neighbors_df : gpd.GeoDataFrame
        neighboring points dataframe with 'geometry' in WKT format.

    Returns
    -------
    dist : dimensions: k x n of distances from origins_df to k neighbors in neighbors_df[idx]

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
    
    
    idx_dist = []
    for tup in list(zip(dist, idx)):
        idx_dist.append(list(zip(tup[1],tup[0])))
        
    
        
    
    #each tuple being (the index of neighbors, the distance to origins)
    return idx_dist
    

def geo_kernel_smoothing(schools_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    

    Parameters
    ----------
    schools_df : gpd.GeoDataFrame
        frame with null values, 'geometry' in WKT format.

    Returns
    -------
    schools_df with null values replaced with average extrapolated from knearest neighbors,
    using distance metric (influence weighted by inverse of distance).
    
    
    Currently this is a bandaid solution for handling null gsRatings.
    
    Would prefer to use multiple imputation, but nearly half the dataset of schools have
    null values, so probably not enough data, but the geographical distribution of
    the null values is dispersed evenly enough with non-null values to allow for a k nearest
    interpolation of values, at least until I implement the review rating aggregation.

    """
    
    # change NoneType to NaN for easier operations
    schools_df['gsRating'].fillna(value=np.nan, inplace=True)
    
    #split into null and not null gsRating values    
    null_schools = schools_df[schools_df['gsRating'].isnull()]        
    neighbor_schools = pd.concat([schools_df, null_schools, null_schools]).drop_duplicates(keep=False)
    
    #grab k nearest
    tuples = geo_knearest(null_schools, neighbor_schools, k=5)
    
    #using inverse distance interpolation to determine neighbor rating weights
    #defaulting to p=2 for inverse squared
    
    
    
    
    
    #using k nearest neighbors for imputation to fill in the NA values
    # imputer = KNNImputer(n_neighbors=5, weights='distance')
    # return imputer.fit_transform(gdf)
    
    return null_schools, neighbor_schools
     
    
    
    
    

def aggregate_school_ratings(properties_df: gpd.GeoDataFrame, schools_df: gpd.GeoDataFrame, kn=3, kavg=5) -> gpd.GeoDataFrame:
    """
    
    Parameters
    ----------
    zillow_df : GeoDataFrame
        Dataframe of housing properties, 'DISTRICT' column with school district 'geometry' column in WKT format
        
    schools_df : GeoDataFrame
        Dataframe of schools, 'geometry' column in WKT format
        
    kn : Integer
        Number of nearest schools for display
        
    kavg : Integer
        Number of nearest schools to calculate average GreatSchools rating

    Returns
    -------
    gdf : GeoDataFrame
        Housing Dataframe with average of the ratings and names of the knearest schools within the same district.

    """
    
    
    
    #first we need to demarcate the properties and schools frame by district, then perform the knearest between each corresponding relation
    #may be computationally inefficient?
    
    #
    properties_grouped = properties_df.groupby('DISTRICT')
    schools_grouped = schools_df.groupby('DISTRICT')
    
    
    
    
    
    

    return None


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




school_districts_df = gpd.read_file('school_districts_bounds/School_District_Boundaries.shp')

#houses with their respective school districts
sjoined_df = assign_property_school_districts(zillow_df, school_districts_df)


tuples = geo_knearest(sjoined_df, schools_df, k=10)




null_schools, neighbors = geo_kernel_smoothing(schools_df)
# print(null_schools.head(), neighbors.head())
print(tuples[0])



# print(dist[0], idx[0])

# for index in idx[0]:
#     print(schools_df.iloc[index])

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












