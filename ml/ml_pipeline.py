import os, sys
import pandas as pd
import numpy as np
import time
import glob
import geopandas as gpd
from shapely.wkt import loads
import time
from geospatial.to_wkt import to_wkt
from geospatial.schoolimputation import assign_property_school_districts, school_imputation, property_school_rating
from ml.kdtree import knearest_balltree
import matplotlib.pyplot as plt
import seaborn as sns
import torch


sns.set(style='whitegrid', palette='muted', font_scale=1)

'''

TODO:
    Pickle:
        multivariate regression model
        the school ratings map
        the crime density map
        also current full database of houses
Once pickled,
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    pass address through:
        use zillow api to add housing attributes
        add schools and location on crime density map
        
    pass address into the model.predict()
    buy or no buy = loaded_model.predict(address)

SEPARATELY:
    we render heatmap of less expensive properties in relation to their
    attributes from current pickled houses object
    
    current_houses = pickle.load(open('houses.pkl', 'rb'))
    


'''

def combine_zillow_csv():
    path = r'/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/houses_compiled/'
    all_files = glob.glob(os.path.join(path, '*.csv'))
    
    
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df = df.drop_duplicates(subset=['zpid'])
    
    #fix the csv write that included indices
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
    df.to_csv('zillow.csv', index=False)
    
def add_schools():
    '''
    read schools
    read districts
    read zillow
    separate all by district, aggregate ratings
    combines, writes to file
    '''
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
    df = pd.read_csv('zillow.csv')
    
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
    
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data')
    zillow.to_csv('fullzillow.csv', index=False)




def normalize_df(df, cols):        
    result = df.copy()
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def rmse_loss(predictions: np.array, targets: np.array):
    '''
    Root Mean Squared Error Loss Function, aka L2 norm
    RMSE = sqrt( avg(y - yhat)^2),
    where y is the observed value and yhat is the prediction.
    
    Measures the average magnitude of error.
    
    RMSE minimizes the squared deviations and finds the *mean*
    MAE minimizes the sum of absolute deviations resulting in the *median*
    
    
    Either we use MAE to handle the outliers, or we divide the dataset
    by zones/clusters, and run RMSE on each zone separately, as the
    model for the Beverly Hills zipcode is going to be essentially useless
    for the model for Northridge, and vice versa.
    
    Might be optimal for this particularly for finding good deals on houses.
    
    '''
    diff = predictions - targets
    diff_squared = np.square(diff)
    mean_diff_squared = np.mean(diff_squared)
    
    rmse = np.sqrt(mean_diff_squared)
    
    return rmse

def mae_loss(predictions, targets):
    '''
    Mean Absolute Loss Error Function, aka L1 norm
    MAE = avg(abs(y - yhat))
    
    Also measures the average magnitude of error, but uses absolute value
    to eliminate the direction of error. It also equally weights all data 
    points. If we run the model on all the zones/clusters together, the
    MAE will probably be optimal, and not be thrown off as much by the
    massive outliers resultant from the wealth disparity in LA.
    
    '''
    diff = predictions - targets
    abs_diff = abs(diff)
    
    mae = np.mean(abs_diff)
    return mae
   
'''
might want to use a combination of both loss functions to handle
bias-variance tradeoff in an optimal manner.
'''

# %% Update dataset
combine_zillow_csv()
add_schools()


# %% Read Zillow data
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data')
zillow = pd.read_csv('fullzillow.csv')
zill_gdf = gpd.GeoDataFrame(zillow, crs='EPSG:4326', geometry=zillow['geometry'].apply(loads))



# %% Read crime density and append to housing dataframe, then log normal of crime
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial')
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

One-hot encoding will be used for the first three codes.

'''

#only keeping first three zip code digits
zips_df = norm_df
zips_df['zipcode'] = zips_df['zipcode'].apply(lambda x: x //100)
dummy_df = pd.get_dummies(zips_df, columns=['useCode', 'zipcode'], drop_first=True, prefix='', prefix_sep='')



# %% Final Cleaning of dataframe of unnecessary columns

'''
Here we drop the unneeded columns for training our model:
    zpid
    percentile
    street
    city
    state
    taxAssessmentYear
    FIPScounty
    lastSoldDate
    lastupdated
    DISTRICT
    school_count
    
    
'''

keep_cols =[col for col in dummy_df.columns if col not in ['zpid', 'percentile', 'street', 'city', 'state', 'taxAssessmentYear',
                                                           'FIPScounty', 'yearBuilt', 'lastSoldDate', 'lastupdated', 'DISTRICT', 
                                                           'school_count']]

X_train = dummy_df[keep_cols]
X_train = X_train.dropna()

# %% last few zero values in rent/zestimate
rentzestimatezero = X_train.index[X_train['rentzestimate']==0]
rentzeros = X_train.loc[rentzestimatezero]

zestimatezero = X_train.index[X_train['zestimate']==0]
zestimatezeros = X_train.loc[zestimatezero]

zeros = rentzeros.append(zestimatezeros)
X_train = X_train.loc[set(X_train.index) - set(zeros.index)]





# %% Supervised Regression

'''
Using mean absolute error loss function

error = mae_loss(predictions, target)

Implement regularisation using lambda parameter.

ie, Loss(y_hat, y) + lambda * N(w),

where w is the weights vector of the loss function, and N(w) is the penalty function,
restricting 
Helps prevent overfitting.

To tune the lambda parameter we use cross-validation: divide training data,
train model for some lambda, test on other half of data. Then repeat procedure
while varying lambda to minimize the loss function.


from torch.nn import Linear
torch.manual_seed(1)
- initializes random seed for our weights

model = Linear(in_features=1, out_features=1)
- for every input feature there is one output features (1 to 1)


'''

num_cols = 43
data_columns = list(X_train.columns)





# %% Define our model
import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        pred = self.linear(x)
        return pred
    
# %% Sample Training Model

X = torch.randn(100, 1)*10
y = X + 3*torch.randn(100, 1)
plt.plot(X.numpy(), y.numpy(), 'o')

plt.ylabel('y')
plt.xlabel('x')


torch.manual_seed(1)
model = LR(1, 1)
print(model)

[w, b] = model.parameters()

def get_params():
    return (w[0][0].item(), b[0].item())

def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    
    x1 = np.array([-30, 30])
    
    #equation of a line
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, y)
    plt.show()

plot_fit('Initial Model')






















