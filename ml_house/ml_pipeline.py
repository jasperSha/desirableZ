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
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

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




def normalize_df(df, cols):        
    result = df.copy()
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# %% Read Zillow data
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml_house')
zillow = pd.read_csv('full_zillow_with_schools.csv')
zill_gdf = gpd.GeoDataFrame(zillow, crs='EPSG:4326', geometry=zillow['geometry'].apply(loads))



# %% Read crime density and append to housing dataframe, then log normal of crime
'''
TODO: clean data such as NaN values, bogus data due to lack of underlying data
      divide based on useCode?
'''

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc')
#crime densities have not been normalized yet
crime_density = pd.read_csv('crime_density_rh_gridsize_1.csv')
cd_gdf = to_wkt(crime_density)

'''
balltree takes 76.64 seconds (maybe needs optimizing)
'''
houses_df = knearest_balltree(zill_gdf, cd_gdf, radius=1.1)
norm_df = houses_df



# %% Use Code Groups
describe = houses_df.describe()

districts = houses_df.groupby('useCode')



condo = districts.get_group('Condominium')
coop = districts.get_group('Cooperative')
duplex = districts.get_group('Duplex')
miscellaneous = districts.get_group('Miscellaneous')
mobile = districts.get_group('Mobile')
familyTwotoFour = districts.get_group('MultiFamily2To4')
familyFivePlus = districts.get_group('MultiFamily5Plus')
quad = districts.get_group('Quadruplex')
townhouse = districts.get_group('Townhouse')
triplex = districts.get_group('Triplex')
unknown = districts.get_group('Unknown')
vacantresidential = districts.get_group('VacantResidentialLand')

# %% Group by Zip Code

zipcodes = houses_df.groupby('zipcode')

beverly_codes = [90035, 90210]
beverly = pd.concat([zipcodes.get_group(code) for code in beverly_codes])

#measuring rent against square footage
x_var = 'finishedSqFt'
data = pd.concat([beverly['rentzestimate'], beverly[x_var]], axis=1)
data.plot.scatter(x=x_var, y='rentzestimate', ylim=(0, 30000))








# %% Single Family Residence Data Cleaning
singlefam = districts.get_group('SingleFamily')


zero_idx = singlefam.index[singlefam['zestimate']==0]
'''
Single Family Residences with a zestimate of 0. 

Last Sold Price maybe more accurate indicator of home value, but for now
we'll just remove the zero valued zestimates. (only 123 of them)
'''
zerosinglefam = singlefam.loc[zero_idx]
singlefam = singlefam.loc[set(singlefam.index) - set(zero_idx)]



# %% Modeling Single Family Data
sns.set(style='whitegrid', palette='muted', font_scale=1)



#beverly hills houses massively skew market, removing for now just for ease visualization
# bev = [90035, 90210]
# singlefam = singlefam[~singlefam['zipcode'].isin(bev)]


describe = singlefam.describe()
# singlefam['rentzestimate'] = np.log(singlefam['rentzestimate'])
sns.distplot(singlefam['lastSoldPrice'])


#model against square footage
x_var = 'finishedSqFt'
data = pd.concat([singlefam['lastSoldPrice'], singlefam[x_var]], axis=1)
data.plot.scatter(x=x_var, y='lastSoldPrice', ylim=(0, 8000000))



#general correlation heatmap
corr_matrix = singlefam.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=.8, square=True)

#check top 5 correlations
k = 10
cols = corr_matrix.nlargest(k, 'rentzestimate')['rentzestimate'].index
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(singlefam[cols].corr(), vmax=.8, square=True)


# %% Checking null data
total = norm_df.isnull().sum().sort_values(ascending=False)
percent = (norm_df.isnull().sum()/norm_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


'''
Missing data:
                   Total   Percent
FIPScounty          2612  0.023418
rentzestimate        147  0.001318
valueChange           32  0.000287
lastSoldDate          10  0.000090
taxAssessmentYear      2  0.000018
lastupdated            1  0.000009
yearBuilt              1  0.000009


This missing data is negligible. Most of it will be cleared with the cleaning
of null values from the single family groupby dataframe.

'''

# %% Cleaning zeros
zero_idx = norm_df.index[norm_df['zestimate']==0]
zeros = norm_df.loc[zero_idx]
norm_df = norm_df.loc[set(norm_df.index) - set(zero_idx)]



# %% Scaling the data

'''
numerical:
crime_density -> log
rentzestimate -> normal
zestimate -> normal
lotSizeSqFt -> normal
low -> normal
high -> normal
zindexValue -> normal
finishedSqFt -> normal
taxAssessment -> normal
edu_rating -> normal
lastSoldDate -> None
yearBuilt -> None
valueChange -> normal


We'll be using normalization instead of standardization. Because the high
variance in the data due to the socioeconomic distribution of Los Angeles,
the data most likely does NOT follow a standard gaussian curve. Subsequently,
normalization makes no assumptions to the distribution of the data, and
allows for the skewed scales, though not to the same degree as using log
to normalize the data.

'''


# log (density + 1) to handle zeros for log norm
norm_df['crime_density'] = norm_df['crime_density'].apply(lambda x: x + 1)
norm_df['crime_density'] = np.log(norm_df['crime_density'])

norm_cols = ['rentzestimate', 'zestimate', 'lotSizeSqFt', 'low', 'high', 'zindexValue',
             'finishedSqFt', 'taxAssessment', 'edu_rating', 'valueChange', 'lastSoldPrice',
             'bathrooms', 'bedrooms']

norm_df = normalize_df(norm_df, norm_cols)




