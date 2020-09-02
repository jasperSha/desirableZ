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

sns.set(style='whitegrid', palette='muted', font_scale=1)

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


def rmse(predictions, targets):
    '''
    Root Mean Squared Error Loss Function
    RMSE = sqrt( avg(y - yhat)^2),
    where y is the observed value and yhat is the prediction.
    
    Measures the average magnitude of error.
    
    Greatly amplifies the error measurement of outliers because of the
    square operation, thus penalizing outliers far more.
    
    Might be optimal for this particular dataset, because of the extreme
    variance in housing values. Either we use RMSE to handle the outliers,
    or, and I think this might be the better option, we divide the dataset
    by zones/clusters, and run models on each zone separately, as the
    model for the Beverly Hills zipcode is going to be essentially useless
    for the model for Northridge.
    
    '''
    diff = predictions - targets
    diff_squared = diff**2
    mean_diff_squared = np.mean(diff_squared)
    
    rmse = np.sqrt(mean_diff_squared)
    
    return rmse

def mae(predictions, targets):
    '''
    Mean Absolute Loss Error Function
    MAE = avg(abs(y - yhat))
    
    Also measures the average magnitude of error, but uses absolute value
    to eliminate the direction of error. It also equally weights all data 
    points. If we separate and run models by zones/clusters, then this
    will probably be the optimal loss function to use.
    
    '''
    diff = predictions - targets
    abs_diff = abs(diff)
    
    mae = np.mean(abs_diff)
    return mae
    
    



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

#convert back to regular dataframe for pandas functions
norm_df = pd.DataFrame(houses_df.drop(columns='geometry'))



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


# %% Modeling Single Family data


singlefam = districts.get_group('SingleFamily')

#Cleaning Zeroes
zero_idx = singlefam.index[singlefam['zestimate']==0]
'''
Single Family Residences with a zestimate of 0. 

Last Sold Price maybe more accurate indicator of home value, but for now
we'll just remove the zero valued zestimates. (only 123 of them)
'''
zerosinglefam = singlefam.loc[zero_idx]
singlefam = singlefam.loc[set(singlefam.index) - set(zero_idx)]



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
print(missing_data)

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
zest_zero_idx = norm_df.index[norm_df['zestimate']==0]
rent_zero_idx = norm_df.index[norm_df['rentzestimate']==0]
zero_idx = zest_zero_idx.append(rent_zero_idx)
zeros = norm_df.loc[zero_idx]
zero_df = norm_df.loc[set(norm_df.index) - set(zero_idx)]




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
bathrooms -> normal
bedrooms -> normal


We'll be using normalization instead of standardization. Because the high
variance in the data due to the socioeconomic distribution of Los Angeles,
the data most likely does NOT follow a standard gaussian curve. Subsequently,
normalization makes no assumptions to the distribution of the data, and
allows for the skewed scales, though not to the same degree as using log
to normalize the data.

Crime density will be log normalized.

zestimate, rentzestimate, taxAssessment, 

'''
norm_df = zero_df

# log (density + 1) to handle zeros for log norm
norm_df['crime_density'] = zero_df['crime_density'].apply(lambda x: x + 1)
norm_df['crime_density'] = np.log(zero_df['crime_density'])

norm_cols = ['rentzestimate', 'zestimate', 'lotSizeSqFt', 'low', 'high', 'zindexValue',
             'finishedSqFt', 'taxAssessment', 'edu_rating', 'valueChange', 'lastSoldPrice',
             'bathrooms', 'bedrooms']

norm_df = normalize_df(norm_df, norm_cols)





# %% Creating dummy variables for useCode categories

'''
In order to account for the different categories of use code,
dummy variables will be created to represent each category.
This is necessary because the evaluations of properties that are
considered "Vacant Residential Lots" are too skewed from "Single Family"
for the Loss function to have any meaning.

To avoid the dummy variable trap, we use SingleFamily as the n0 variable,
and for the rest we'll use one-hot encoding.

Coding for the rest of the dummy variables:
    for (0 to 11) di to dn:
        d0 : 'Condominium'
        d1 : 'Cooperative'
        d2 : 'Duplex'
        d3 : 'Miscellaneous'
        d4 : 'Mobile'
        d5 : 'MultiFamily2To4'
        d6 : 'MultiFamily5Plus'
        d7 : 'Quadruplex'
        d8 : 'Townhouse'
        d9 : 'Triplex'
        d10 : 'Unknown'
        d11 : 'VacantResidentialLand'
        
Zipcodes will also be categorical. However, due to the structure of the zipcode,
there can be a bit more information gleaned from it.

The first digit represents the state. 2nd and 3rd represent the sectional center
or large city post office. Final two digits represent the associate post office
or delivery area. Essentially zip codes do NOT represent geographical areas,
but address groups/delivery routes. Thus they can geographically overlap.


Although the first digit isn't necessarily relevant now, as all our test data
is contained within Los Angeles, CA, we'll keep it for sake of posterity/scaleability.
We'll keep the 2nd and 3rd as well for denotation of a decently sized distribution 
zone, and we'll leave the last two so that our granularity is not TOO fine.

So the first three will be treated as categorical, and dummy variables will
be created for them.


    
'''

#get_dummies automatically unravels column and assigns value by one-hot encoding
#Condominium was dropped as the dummy variable trap, and useCode was dropped as well.
dummy_df = pd.get_dummies(norm_df, columns=['useCode'], drop_first=True, prefix='', prefix_sep='')



# %% Final Cleaning of dataframe of unnecessary columns

'''
Here we drop the unneeded columns for training our model:
    zpid
    percentile
    street
    city
    state
    taxAssessmentYear
    
    
'''








