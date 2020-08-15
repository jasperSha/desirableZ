
import os, sys
import pandas as pd
import glob
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
# from geospatial_calc.schoolimputation import testing, assign_property_school_districts, school_imputation, geo_knearest, property_school_rating

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
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    df = df.drop(['longitude', 'latitude'], axis=1)
    
    zillow_gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=geometry)

    
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
    



os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml_house')
zillow = pd.read_csv('full_zillow_with_schools.csv')


# #next add crime density variable

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc')

crime_density = pd.read_csv('crime_density_rh_gridsize_1.csv')



'''
TODO: use geoknearest + adapt the point density mesh to append crime densities to each house
'''






#begin modeling


















