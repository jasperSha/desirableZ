import pandas as pd
import numpy as np
import geopandas as gpd
from ml_house.kdtree import geo_knearest


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








