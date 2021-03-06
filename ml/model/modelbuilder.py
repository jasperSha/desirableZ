import os, sys
import time
import glob
import math
import joblib
import re
import gc

import geopandas as gpd
from shapely.wkt import loads
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from geospatial.more_columns import prox_transpo, prox_colleges, prox_worship, proximal_locations
from geospatial.to_wkt import to_wkt
from geospatial.schoolimputation import assign_property_school_districts, school_imputation, property_school_rating
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
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'amount', 'last-updated'], axis=1)
    
    '''
    TODO:
        READ ZINDEXVALUE IN AS A FLOAT64, REMOVE QUOTES AND COMMA FROM FIELDS
    '''
    
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
    
    #for random sampling   
    # df = df.sample(n=50)
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
    # zillow.to_csv('random_sample_5000_zillow.csv', index=False)



def normalize_df(df, cols):        
    result = df.copy()
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def update_data():
    # Update dataset
    combine_zillow_csv()
    add_schools()
    
# %%

def assign_dtypes():
    cd = os.getcwd()
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data')
    dtypes = { 'FIPScounty' : 'float64',
               'bathrooms' : 'float64',
               'bedrooms' : 'float64',
               'city' : object,
               'finishedSqFt' : 'float64',
               'high' : 'float64',
               'lastSoldDate' : object,
               'lastSoldPrice' : 'float64',
               'lastupdated' : object,
               'lotSizeSqFt' : 'float64',
               'low' : 'float64',
               'percentile' : 'float64',
               'rentzestimate' : 'float64',
               'state' : object,
               'street' : object,
               'taxAssessment' : 'float64',
               'taxAssessmentYear' : object,
               'useCode' : object,
               'valueChange' : 'float64',
               'yearBuilt' : object,
               'zestimate' : 'float64',
               'zindexValue' : object,
               'zipcode' : 'int64',
               'zpid' : object,
               'last-updated' : object,
               'geometry' : object,
               'DISTRICT' : object,
               'edu_rating' : 'float64',
               'school_count' : 'int64'        
            }
    
    #remove commas, convert zindexValue to int64
    df = pd.read_csv('fullzillow.csv', dtype=dtypes)
    df['zindexValue'] = df['zindexValue'].apply(lambda x: x.replace('"', ''))
    df['zindexValue'] = df['zindexValue'].apply(lambda x: x.replace(',', ''))
    df['zindexValue'] = pd.to_numeric(df['zindexValue'])
    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=df['geometry'].apply(loads))
    os.chdir(cd)
    return gdf



# %% Recompile Zillow data
# combine_zillow_csv()

# Schools
# add_schools()

zill_gdf = assign_dtypes()


# %% Building sample dataset for map

df = zill_gdf
print(df.columns)
print(df.shape)
df = df.loc[df['zestimate'] != 0]
df = df.loc[df['useCode'] == 'SingleFamily']
print(df.shape)
df = df.dropna(subset=['zestimate'])


zipcodes_df = df.groupby('zipcode')
zipcodes_df = zipcodes_df.apply(lambda x: x.sample(frac=0.1))

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data')

zipcodes_df.to_csv('sample_zipcodes_0.1_zestimate.csv', index=False)



# %% Read Zillow data

zill_gdf = assign_dtypes()
zill_gdf = proximal_locations(zill_gdf, flag='transpo')
zill_gdf = proximal_locations(zill_gdf, flag='colleges')
zill_gdf = proximal_locations(zill_gdf, flag='worship')


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

#drop NaN results
zero_df = zero_df.dropna(subset=['rentzestimate', 'zestimate'])

# %% Scaling the data

norm_df = zero_df

# log (density + 1) to handle zeros for log norm
norm_df['crime_density'] = zero_df['crime_density'].apply(lambda x: x + 1)
norm_df['crime_density'] = np.log(zero_df['crime_density'])

x_scaler = MinMaxScaler()
y_zest_scaler = MinMaxScaler()
y_rent_scaler = MinMaxScaler()

x_norm_cols = ['lotSizeSqFt', 'finishedSqFt', 'taxAssessment', 'edu_rating', 
             'valueChange', 'bathrooms', 'bedrooms', 'yearBuilt',
             'transpo_prox', 'colleges_prox', 'worship_prox']

#scaling our data for dependent and independent
norm_df[x_norm_cols] = x_scaler.fit_transform(norm_df[x_norm_cols])

#keeping scalers for rent and zestimate separate
zest_norm_col = ['zestimate']
rent_norm_col = ['rentzestimate']

norm_df[zest_norm_col] = y_zest_scaler.fit_transform(norm_df[zest_norm_col])
norm_df[rent_norm_col] = y_rent_scaler.fit_transform(norm_df[rent_norm_col])

# %% Preserve Scalers, Column order
cwd = os.getcwd()
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')

#save scalers for user pipeline
joblib.dump(x_scaler, 'x_scaler.gz')
joblib.dump(y_zest_scaler, 'y_zest_scaler.gz')
joblib.dump(y_rent_scaler, 'y_rent_scaler.gz')

#save columns for user pipeline
joblib.dump(x_norm_cols, 'x_cols.pkl')
joblib.dump(zest_norm_col, 'zest_col.pkl')
joblib.dump(rent_norm_col, 'rent_col.pkl')

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
zone, and we'll remove the last two so that our granularity is not TOO fine.

One-hot encoding will be used for the first three codes.

'''

# #only keeping first three zip code digits
# zips_df = norm_df
# zips_df['zipcode'] = zips_df['zipcode'].apply(lambda x: x //100)
# dummy_df = pd.get_dummies(zips_df, columns=['useCode', 'zipcode'], drop_first=True, prefix='', prefix_sep='')

#trying only for usecode = 'SingleFamily'
zips_df = norm_df
zips_df['zipcode'] = zips_df['zipcode'].apply(lambda x: x //100)
zips_df = zips_df.loc[zips_df['useCode'] == 'SingleFamily']
dummy_df = pd.get_dummies(zips_df, columns=['zipcode'], drop_first=True, prefix='', prefix_sep='')


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
    zindexValue
    valueChange
    low
    high    
    
    testing:
        keep: yearBuilt
        remove: useCode,
                crime_density
        
    
    
'''

x_df = dummy_df.drop(columns=['zpid', 'percentile', 'street', 'city', 'state', 'taxAssessmentYear',
                              'FIPScounty', 'lastSoldDate', 'lastupdated', 'last-updated', 'DISTRICT', 
                              'school_count', 'zindexValue', 'valueChange', 'low', 'high', 'taxAssessment',
                              'useCode', 'crime_density'])




x_df = x_df.dropna()

# last few zero values in rent/zestimate
rentzestimatezero = x_df.index[x_df['rentzestimate']==0]
rentzeros = x_df.loc[rentzestimatezero]

zestimatezero = x_df.index[x_df['zestimate']==0]
zestimatezeros = x_df.loc[zestimatezero]

zeros = rentzeros.append(zestimatezeros)
x_df = x_df.loc[set(x_df.index) - set(zeros.index)]





# %% Split Independent and Dependent

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
y_zest_col = ['zestimate']

#saving rent column
y_rent_col = ['rentzestimate']


y = pd.DataFrame(x_df, columns=y_zest_col)
x = x_df.drop(columns=['rentzestimate', 'zestimate'])

x, y = x.astype(np.float32), y.astype(np.float32)


# %% Preserve order of columns
predictor_cols = x.columns

cwd = os.getcwd()
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
joblib.dump(predictor_cols, 'predictor_cols_03.pkl')
os.chdir(cwd)


# %% Convert to Tensors, Set Hyperparameters

x_tensor = None
y_tensor = None

#note: use from_numpy(), using torch.tensor does conversion inplace, inconvenient for testing
x_tensor = torch.from_numpy(x.values)
y_tensor = torch.from_numpy(y.values)

#43 columns
D_in, D_out = x_tensor.shape[1], y_tensor.shape[1]

#Hyperparameters
lr = .0005 # optimal learning rate for batch size of 64
# lr = .03 # optimal learning rate for batch size 32
epochs = 100
L1, L2, L3, L4 = 4000, 5000, 5000, 4000


# %%Init model, optimizer
model = Net(D_in, D_out, L1, L2, L3, L4)

#create device object for cuda operations
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#move to GPU (this must be done before setting any optimizers)
model.to(dev)

#default reduction is 'mean' - loss is then independent of batch size
criterion = nn.MSELoss()

# #less tilted by outliers with mean absolute error func (this one's awful)
# criterion = nn.L1Loss()

# #set our optimizer=Adam
# opt = torch.optim.Adam(model.parameters(), lr=lr)

# #AdamW, default parameters (lr=1e-3, weight_decay=1e-2, eps=1e-8)   (best so far)
# opt = torch.optim.AdamW(model.parameters(), lr=lr)
'''
result: Epoch: 100 loss: 0.9383808509883238 validation loss: 0.21503481738909613
'''

#AdamW, default params, using AMSGrad variant
opt = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True) #just added manual garbage cleanup after training
'''
result: (lr=.0001, neurons: 200, 300, 300, 200, batch=128, drop useCode): 
    Epoch: 100 loss: 37.06490480247885 validation loss: 7.978232683613896
    
result: (lr=.0005, neurons: 200, 300, 300, 200, batch=128, drop useCode, crime_density): 
    Epoch: 100 loss: 0.6585311957169324 validation loss: 0.1456557799683651

result: (lr=.0005, neurons: 400, 500, 500, 400, batch=128, drop useCode, crime_density): 
    Epoch: 100 loss: 0.5185779714738601 validation loss: 0.11540724463702645
    
result: (lr=.0005, neurons: 4000, 5000, 5000, 4000, batch=128, drop useCode, crime_density):  
    Epoch: 100 loss: 0.3578301961533725 validation loss: 0.08210584594235115   (long-ish runtime, possibly could run more epochs
                                                                                               for better loss)
    
result: (lr=.0005, neurons: 5000, 6000, 6000, 5000, batch=64, drop useCode, crime_density):  
    Epoch: 168 loss: 0.9562640792391903 validation loss: 0.21707699820763082 (incomplete, ran out of memory
                                                                              aiming for 200 epochs)
'''

# %% Split and Set DataLoader

#train and validation sets
X_train, X_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=20)


#load datasets for iterator and set our batch sizes
train_set = torch.utils.data.TensorDataset(X_train, y_train)

validation_set = torch.utils.data.TensorDataset(X_test, y_test)

train_params = {'batch_size' : 128,
                'shuffle' : True,
                'pin_memory' : True,
                'num_workers' : 0}

test_params = {'batch_size' : 128,
               'shuffle' : False}

#training set
train_iter = torch.utils.data.DataLoader(train_set, **train_params)

#testing set
validation_iter = torch.utils.data.DataLoader(validation_set, **test_params)


# %%  Training Loop
torch.backends.cudnn.benchmark=True

#resetting iterators for tests
x = None
y = None
losses = []
val_losses = []

running_loss = 0.0
validation_loss = 0.0

y_hat, y_actual = [], []


for epoch in range(epochs):
    for x, y in train_iter:
        
        #set model to training mode (default)
        model.train()

        #transfer batch to gpu
        x, y = x.to(dev), y.to(dev)
        
        #clear gradients
        opt.zero_grad()
        
        #make prediction
        y_pred = model(x)
        
        #storing the original vs predictions
        y_actual.append(x)
        y_hat.append(y_pred)
        
        loss = criterion(y_pred, y)
        
        #backpropagation
        loss.backward()
        
        #apply gradients found to our optimizer
        opt.step()
        
        #store loss
        running_loss += loss.item()
        losses.append(running_loss)
        
    with torch.no_grad():
        for x_val, y_val in validation_iter:
            
            model.eval()
            x_val, y_val = x_val.to(dev), y_val.to(dev)
            
            y_pred = model(x_val)
            
            val_loss = criterion(y_pred, y_val)
            
            validation_loss += val_loss.item()
            
            #storing to evaluate
            val_losses.append(validation_loss)
            
        
    print('Epoch:', epoch+1, 'loss:', running_loss, 'validation loss:', validation_loss)
    running_loss = 0.0
    validation_loss = 0.0

# %% ensure memory deallocation
del model, opt
gc.collect()
torch.cuda.empty_cache()


# %% Query Nvidia-smi
import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


# %%


y_actual = y_rent_scaler.inverse_transform(np.reshape(y_actual, (-1, 4)))

y_hat = [x.cpu() for x in y_hat]
y_hat = [x.numpy() for x in y_hat]
y_hat = y_rent_scaler.inverse_transform(np.reshape(y_hat, (-1, 4)))






# %%
'''
need to save D_in, D_out, predictor_cols, model

'''
model.eval()




# %% Compile model for use in Production

model.eval()
traced_model = torch.jit.trace(model, )

# %% Saving the model

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
FILENAME = 'state_dict_model_03.pt'

torch.save(model.state_dict(), FILENAME)


# %% Extra Comparisons
    
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

