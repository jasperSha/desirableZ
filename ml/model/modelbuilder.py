import os, sys
import time
import glob
import math
import joblib

import geopandas as gpd
from shapely.wkt import loads
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from geospatial.to_wkt import to_wkt
# from geospatial.schoolimputation import assign_property_school_districts, school_imputation, property_school_rating
from ml.kdtree import knearest_balltree
from ml.model.neuralnet import Net


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
    we render heatmap of randomly picked, but spread out properties,
    with all their features as well as crime density ratings/education ratings,
    and then color gradient in terms of their under/overvalued rating
    designated by the regression model. we pass these houses using
    json to react in order to then mark it onto the mapbox map.
    
    
    the mapbox map will have a side panel that pops up with all the information
    about the house as you hover over it, and you can enter new houses
    to see how they rate, and see it pop up onto the map in realtime.
    
    
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




# %%
    
def check_collinearity(): 
    # Use Code Groups
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
    
    
    #Modeling Single Family data
    
    
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
    
    
    
    
    #Checking null data
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
def update_data():
    # Update dataset
    combine_zillow_csv()
    add_schools()
    
    

# %% Read Zillow data

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data')
zillow = pd.read_csv('fullzillow.csv')
zill_gdf = gpd.GeoDataFrame(zillow, crs='EPSG:4326', geometry=zillow['geometry'].apply(loads))



# %% Read crime density and append to housing dataframe, then log normal of crime
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data')
#crime densities have not been normalized yet
crime_density = pd.read_csv('crime_density_rh_gridsize_1.csv')
cd_gdf = to_wkt(crime_density)

'''
balltree takes 76.64 seconds (maybe needs optimizing)
'''
houses_df = knearest_balltree(zill_gdf, cd_gdf, radius=1.1)

#convert back to regular dataframe for pandas functions
norm_df = pd.DataFrame(houses_df.drop(columns='geometry'))


# %% Cleaning zeros and the Vacant Lots/Unknown use codes
zest_zero_idx = norm_df.index[norm_df['zestimate']==0]
rent_zero_idx = norm_df.index[norm_df['rentzestimate']==0]

unknown = norm_df.index[norm_df['useCode']=='Unknown']
vacant = norm_df.index[norm_df['useCode']=='VacantResidentialLand']

unwanted_idx = zest_zero_idx.append([rent_zero_idx, unknown, vacant])
zero_df = norm_df.loc[set(norm_df.index) - set(unwanted_idx)]




# %% Scaling the data

'''
numerical:
crime_density -> log
rentzestimate -> None (output)
zestimate -> None (output)
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

MinMaxScaler can be saved to apply to future data/tests.


'''
norm_df = zero_df

# log (density + 1) to handle zeros for log norm
norm_df['crime_density'] = zero_df['crime_density'].apply(lambda x: x + 1)
norm_df['crime_density'] = np.log(zero_df['crime_density'])

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

x_norm_cols = ['lotSizeSqFt', 'low', 'high', 'zindexValue',
             'finishedSqFt', 'taxAssessment', 'edu_rating', 
             'valueChange', 'lastSoldPrice','bathrooms', 'bedrooms']

y_norm_cols = ['rentzestimate', 'zestimate']

#create separate scalers for our dependent and independent variables
norm_df[x_norm_cols] = x_scaler.fit_transform(norm_df[x_norm_cols])
norm_df[y_norm_cols] = y_scaler.fit_transform(norm_df[y_norm_cols])

cwd = os.getcwd()
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')

joblib.dump(x_scaler, 'x_scaler.gz')
joblib.dump(y_scaler, 'y_scaler.gz')

joblib.dump(x_norm_cols, 'x_cols.pkl')
joblib.dump(y_norm_cols, 'y_cols.pkl')

os.chdir(cwd)

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





# %% Convert to Tensors

'''

To tune the lambda parameter we use cross-validation: divide training data,
train model for some lambda, test on other half of data. Then repeat procedure
while varying lambda to minimize the loss function.

Using MSE loss instead of RMSE to prevent infinity output when autograd performs 
the backward pass and attempts to derive sqrt at 0.

We're using ReLu for our activation functions:
    allows neurons to demonstrate stronger activation(whereas sigmoid the difference
                                                      between a relatively weak 
                                                      activation and a strong activation
                                                      is more impactful)
    less sensitive to random intiialization, vs sigmoid/tanh
    easier to derive for our gradient descent
    
But for our output layer, we will use a linear activation function, as we are
predicting a continuous value. So we don't want to be restricted in terms of
the possible value outputs, we want the whole range to get the real difference
between predicted/target.

Stochastic Gradient Descent (maybe we can try RMSProp)


'''


#using just the zestimate, or the Zillow estimated home value as our target variable 
y_col = ['zestimate']
y = pd.DataFrame(X_train, columns=y_col)
x = X_train.drop(['rentzestimate', 'zestimate'], axis=1)

#preserve order of columns
predictor_cols = x.columns

cwd = os.getcwd()
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
joblib.dump(predictor_cols, 'predictor_cols.pkl')
os.chdir(cwd)

# %% Convert to Tensors

x = torch.tensor(x.values, dtype=torch.float)
y = torch.tensor(y.values, dtype=torch.float)




# %%

#43 columns
D_in, D_out = x.shape[1], y.shape[1]
lr = 1e-2
epochs = 1000

#init model
model = Net(D_in, D_out)

#create device object for cuda operations
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#finally we move the model to the GPU (this must be done before setting any optimizers)
model.to(dev)

#train and validation sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

#default reduction is 'mean' - loss is then independent of batch size
criterion = nn.MSELoss()

#set our optimizer=stochastic gradient descent
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#load datasets for iterator and set our batch sizes
train_set = torch.utils.data.TensorDataset(X_train, y_train)
validation_set = torch.utils.data.TensorDataset(X_test, y_test)

train_params = {'batch_size' : 64,
                'shuffle' : True,
                'num_workers' : 0}

test_params = {'batch_size' : 64,
               'shuffle' : False}

train_iter = torch.utils.data.DataLoader(train_set, **train_params)
validation_iter = torch.utils.data.DataLoader(validation_set, **test_params)

losses = []
val_losses = []
running_loss = 0.0

for epoch in range(epochs):
    for x, y in train_iter:
        
        #transfer batch to gpu
        x, y = x.to(dev), y.to(dev)
        
        #clear gradients
        opt.zero_grad()
        
        #make prediction
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        #backpropagation
        loss.backward()
        
        #apply gradients found to our optimizer
        opt.step()
        
        running_loss += loss.item()
    print('Epoch:', epoch+1, 'loss:', running_loss)
    running_loss = 0.0
        

# for x, y in validation_iter:
#     print(x, y)

# with torch.no_grad():
#     for epoch in range(epochs):
#         for x_test, y_test in validation_iter:
#             x_test, y_test = x_test.to(dev), y_test.to(dev)
            
#             y_test_pred = model(x_test)
#             loss = criterion(y_test_pred, y_test)
            
#             print('Actual:', y_test.item(), '\n', 'Predicted:', y_test_pred.item())
#             print('loss:',loss.item(), '\n')
            

# %%
'''
need to save D_in, D_out, predictor_cols, model

'''


print(x.shape)

print(x[0])



# %% Compile model for use in Production

model.eval()
traced_model = torch.jit.trace(model, )

# %% Saving the model

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
PATH = 'state_dict_model.pt'

torch.save(model.state_dict(), PATH)


