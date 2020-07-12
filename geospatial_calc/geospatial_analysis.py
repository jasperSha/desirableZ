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
    schools_joined_districts_df : properties_df, each with their respective school district
        

    """
      
    schools_joined_districts_df = gpd.sjoin(properties_df, districts_df, op='within')
    
    #preserve property columns, just concat the school districts column-wise
    columns = properties_df.columns.values.tolist()
    columns.append('DISTRICT')
    
    schools_joined_districts_df = schools_joined_districts_df[columns]
    
    return schools_joined_districts_df




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
        property dataframe with 'geometry' in WKT format.
    neighbors_df : gpd.GeoDataFrame
        schools in the same district as the properties dataframe with 'geometry' in WKT format.

    Returns
    -------
    2-tuple: (gsID, distance) of neighboring k=10 schools to each property in origins_df (in the same district)

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
    gsid = []
    for school_idx in idx:
        gsid.append(neighbors_df['gsId'].iloc[school_idx])
    gsratings = []
    #can't do ratings here until we either impute it or once we get the school api key
    for ratings in idx:
        gsratings.append(neighbors_df['gsRating'].iloc[ratings].astype('int64'))
        
    
    id_ratings_dist = []
    for tup in list(zip(gsid, gsratings, dist)):
        id_ratings_dist.append(list(zip(tup[0],tup[1], tup[2])))
        
    
    return id_ratings_dist
    

    
    
    
    




def school_imputation(schools_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    
    Ratings interpolation is done across and in spite of district lines, as funding for schools,
    at least in California, is decided at a state level. This does not take into account
    the differences between public and charter schools. Though the ratings imputation with
    review aggregation will somewhat make up for the difference, in all honesty quite a few more
    refinements are needed to truly handle accurate judgments of education quality.

    """
    
    # change NoneType to NaN for easier operations
    schools_df['gsRating'].fillna(value=np.nan, inplace=True)
    
    #split into null and not null gsRating values dataframes    
    null_schools = schools_df[schools_df['gsRating'].isnull()]        
    neighbor_schools = pd.concat([schools_df, null_schools, null_schools]).drop_duplicates(keep=False)
    
    #grab k nearest
    id_ratings_dist = geo_knearest(null_schools, neighbor_schools, k=5)
    
    
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

schools_df = gpd.read_file('greatschools/joined.shp')

#school district boundaries
school_districts_df = gpd.read_file('school_districts_bounds/School_District_Boundaries.shp')

#houses with their respective school districts
schools_joined_districts_df = assign_property_school_districts(zillow_df, school_districts_df)



'''
right now just checking how to grab the k nearest schools and then their ratings. once that's accomplished,
we then figure out the aggregate_school_ratings function to split on districts, and apply
geo_knearest on each property/school district.
'''
# tuples = geo_knearest(schools_joined_districts_df, schools_df, k=10)




# null_schools, neighbors = school_imputation(schools_df)
# print(null_schools.head(), neighbors.head())









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












