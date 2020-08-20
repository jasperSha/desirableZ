import os, sys
import pandas as pd
import numpy as np
import time
import glob
import geopandas as gpd
from shapely.wkt import loads
import time
from geospatial_calc.to_wkt import to_wkt
from kdtree import knearest_balltree
import matplotlib.pyplot as plt

'''
current bottlenecks:
    school imputation
    crime_df compiling (point density algo itself is fast enough, need to write full crime_df to csv)

'''

def combine_zillow_csv():
    path = r'/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/houses_compiled/'
    all_files = glob.glob(os.path.join(path, '*.csv'))
    
    
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    
    df = df.drop_duplicates(subset=['zpid'])
    df = df.drop(df.columns[[0, 1]], axis=1)
    new_cols = ['zpid', 'rentzestimate', 'zestimate', 'low', 'high', 'valueChange',
                'zindexValue', 'percentile', 'street', 'city', 'state', 'lotSizeSqFt', 'finishedSqFt',
                'taxAssessment', 'taxAssessmentYear','zipcode', 'useCode', 'yearBuilt',
                'bathrooms', 'bedrooms', 'FIPScounty', 'lastSoldDate', 'lastSoldPrice',
                'lastupdated', 'latitude', 'longitude']
    df = df[new_cols]
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml_house')
    df.to_csv('full_zillow.csv', index=False)
    
def add_schools():
    '''
    read schools
    read districts
    read zillow
    separate all by district, aggregate ratings
    '''
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml_house')
    df = pd.read_csv('full_zillow.csv')
    
    zillow_gdf = to_wkt(df)

    
    os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
    schools_df = gpd.read_file('greatschools/joined.shp')
    schools_df.crs = 'EPSG:4326'
    #school district boundaries
    school_districts_df = gpd.read_file('school_districts_bounds/School_District_Boundaries.shp')
    school_districts_df.crs = 'EPSG:4326'
    #houses with their respective school districts
    zill_df = assign_property_school_districts(zillow_gdf, school_districts_df)
    
    
    #split properties and rated schools into separate dataframes by district, then match aggregate school ratings
    full_schools = school_imputation(schools_df)
    zillow = property_school_rating(zill_df, full_schools)
    
    return zillow
    

def normalize(df):
    df = df.select_dtypes(include=['int64', 'float64'])
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result



os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml_house')
zillow = pd.read_csv('full_zillow_with_schools.csv')
zill_gdf = gpd.GeoDataFrame(zillow, crs='EPSG:4326', geometry=zillow['geometry'].apply(loads))


# #next add crime density variable

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc')

#crime densities have not been normalized yet
crime_density = pd.read_csv('crime_density_rh_gridsize_1.csv')
cd_gdf = to_wkt(crime_density)


start = time.time()

#add crime densities to houses
'''
balltree takes 76.64 seconds (maybe needs optimizing)
'''
houses_df = knearest_balltree(zill_gdf, cd_gdf, radius=1.1)

end = time.time()
print('balltree time: ', end - start)



#log (density + 1) to handle zeros for log norm
houses_df['crime_density'] = houses_df['crime_density'].apply(lambda x: x + 1)
houses_df['crime_density'] = np.log(houses_df['crime_density'])


'''
TODO: clean data such as NaN values, bogus data due to lack of underlying data
      divide based on useCode?


'''
# houses_df = normalize(houses_df)

#take a look at the data
describe = houses_df.describe()

districts = houses_df.groupby('DISTRICT')
print(districts[0])





#begin modeling


















