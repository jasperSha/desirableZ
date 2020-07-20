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


#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None
#turn off chained assignment on dataframe alert
pd.options.mode.chained_assignment = None




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

     
def split_apply_combine(properties_df: gpd.GeoDataFrame, schools_df: gpd.GeoDataFrame, kavg: int=5) -> gpd.GeoDataFrame:
    """
    
    Parameters
    ----------
    zillow_df : GeoDataFrame
        Dataframe of housing properties, 'DISTRICT' column with school district 'geometry' column in WKT format
        
    schools_df : GeoDataFrame
        Dataframe of schools, 'geometry' column in WKT format
        
    kavg : Integer
        Number of nearest schools used to calculate average GreatSchools rating

    Returns
    -------
    gdf : GeoDataFrame
        Housing Dataframe with average of the ratings and names of the knearest schools within the same district.

    """
    
    
    
    #first we need to demarcate the properties and schools frame by district, then perform the knearest between each corresponding relation
    #may be computationally inefficient?
    
    #
    properties = properties_df.groupby('DISTRICT')
    schools = schools_df.groupby('DISTRICT')
    
    
    
    
    
    
    

    return properties, schools

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

awsz_df = gpd.read_file('zillowdb/zillowaws.shp')
awsz_df.drop_duplicates(subset=['street', 'city'],inplace=True)


schools_df = gpd.read_file('greatschools/joined.shp')

#school district boundaries
school_districts_df = gpd.read_file('school_districts_bounds/School_District_Boundaries.shp')

#houses with their respective school districts
zill_df = assign_property_school_districts(zillow_df, school_districts_df)




# tuples = geo_knearest(schools_joined_districts_df, schools_df, k=10)



#split properties and rated schools into separate dataframes by district, then match aggregate school ratings
full_schools = school_imputation(schools_df)


'''
first: get list of all the districts the properties are in.
second: group the schools by districts
third: iterate through each property (individuals), feed that into function with corresponding
       schools frame to find knn
fourth: aggregate ratings.



'''






def property_school_rating(zill_df: gpd.GeoDataFrame, full_schools: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    props, schols = split_apply_combine(zill_df, full_schools)

    property_districts = zill_df['DISTRICT'].unique()
    
    schol_distr = [schols.get_group(x) for x in schols.groups]
    prop_distr = [props.get_group(x) for x in props.groups]


    #grab dataframe separated by district
    for house_district_group in prop_distr:
        #initialize education quality column
        house_district_group['edu_rating'] = np.nan
        
        
        district = house_district_group['DISTRICT'].unique()[0]
        schools = schols.get_group(district)
        
        #get neighboring schools in the same district
        zp_id_dist = geo_knearest(house_district_group, schools, impute=False, k=10)
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
            house_district_group['edu_rating'].loc[house_district_group['zpid']==house] = weighted_rating
        print(house_district_group)
        break



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












