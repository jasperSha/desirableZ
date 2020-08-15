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
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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



def geo_knearest(origins_df: gpd.GeoDataFrame, neighbors_df: gpd.GeoDataFrame, impute: bool=True, k: int=10) -> list:
    """
    

    Parameters
    ----------
    origins_df : gpd.GeoDataFrame
        property dataframe with 'geometry' in WKT format.
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
    
        


def school_imputation(schools_df: gpd.GeoDataFrame, k: int=3) -> gpd.GeoDataFrame:
    """
    

    Parameters
    ----------
    schools_df : gpd.GeoDataFrame
        frame with null values, 'geometry' in WKT format.

    Returns
    -------
    schools_df with null values replaced with average extrapolated from knearest neighbors,
    using distance metric (influence weighted by inverse of distance).
    
    Diff methods:
        1. Multi-variate chained equation (MICE) - dependent on null being missing at random (MAR)
        2. K-NN + mean imputation (sensitive to outliers)
        3. Do nothing, leave as null, aggregate ratings where possible
        
    Going to use KNN + mean imputation combined with inverse distance weighting; the dataset has little 
    to no outliers from inspection and MICE is dependent on multiple variables for linear regression 
    for each step, and these schools only have the one. A caveat is that private/charter schools receive 
    separate funding from public schools, and this can have a non-negligible effect on their quality.
    

    """
    
    # change NoneType to NaN for easier operations
    schools_df['gsRating'].fillna(value=np.nan, inplace=True)
    
    #split into null and not null gsRating values dataframes    
    null_schools = schools_df[schools_df['gsRating'].isnull()]        
    rated_schools = pd.concat([schools_df, null_schools, null_schools]).drop_duplicates(keep=False)
    
    #grab k nearest
    id_dist = geo_knearest(null_schools, rated_schools, k=k)
    
    # #using inverse distance interpolation to determine neighbor rating weights
    for null_index, neighbors in enumerate(id_dist):
        #replace gsIDs with respective ratings
        school_ids = [x[0] for x in neighbors]
        ratings = []
        for gsId in school_ids:
            school = rated_schools.loc[rated_schools['gsId']==gsId]['gsRating']
            rating = school.iloc[0]
            ratings.append(float(rating))
        
        distances = np.array([x[1] for x in neighbors])
        
        #to handle schools whose gps coordinates located in the same spot, scale all by 1 to avoid divide by zero
        distances += 1
        
        #normalize distance weights
        weights = 1/distances
        weights /= weights.sum(axis=0)
        ratings *= weights.T
        
        weighted_rating = round(ratings.sum(axis=0))
        
        #replace null school with imputed value
        null_schools.at[null_index, 'gsRating'] = weighted_rating
    full_schools = pd.concat([null_schools, rated_schools], ignore_index=True)
    
    return full_schools

     

def property_school_rating(zill_df: gpd.GeoDataFrame, full_schools: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    #init school column for main frame
    zill_df['edu_rating'] = np.nan
    
    houses, schols = zill_df.groupby('DISTRICT'), full_schools.groupby('DISTRICT')
    
    # property_districts = zill_df['DISTRICT'].unique()
    
    # schol_distr = [schols.get_group(x) for x in schols.groups]
    house_districts = [houses.get_group(x) for x in houses.groups]


    #grab dataframe separated by district
    for house_district_group in house_districts:
        #initialize education quality column
        # house_district_group['edu_rating'] = np.nan
        
        
        district = house_district_group['DISTRICT'].unique()[0]
        schools = schols.get_group(district)
        
        k = len(schools) if len(schools) < 5 else 5
        
        #get neighboring schools in the same district
        zp_id_dist = geo_knearest(house_district_group, schools, impute=False, k=k)
        for house, neighboring_schools in zp_id_dist.items():
            #replace gsIDs with respective ratings
            school_ids = [x[0] for x in neighboring_schools]
            ratings = []
            for gsId in school_ids:
                school = schools.loc[schools['gsId']==gsId]['gsRating']
                rating = school.iloc[0]
                ratings.append(float(rating))
            
            distances = np.array([x[1] for x in neighboring_schools])
            
            #to handle schools whose gps coordinates located in the same spot, scale all by 1 to avoid divide by zero
            distances += 1
            
            #normalize distance weights
            weights = 1/distances
            weights /= weights.sum(axis=0)
            ratings *= weights.T
            
            weighted_rating = ratings.sum(axis=0)
            
            #append the aggregate rating to associated housing property        
            zill_df.at[zill_df['zpid']==house, 'edu_rating'] = weighted_rating
            zill_df.at[zill_df['zpid']==house, 'school_count'] = k
            # zill_df['edu_rating'].loc[zill_df['zpid']==house] = weighted_rating
    return zill_df








