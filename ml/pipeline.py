import os
import joblib
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from deepsearchresults import deep_search
from ml.model.neuralnet import Net
from zillowObject.zillowObject import House

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ')
key = os.getenv('ZILLOW_API_KEY')


# %%
'''
pipe:
    enter address
    use zestimate + deepsearch to get zillow values
    add schools to frame
    add crime density to frame
    x_scaler.transform(input[independent_variables])
    y_scaler.transform(input[dependent_variables])
    
    y_pred = model(x_scaler)
    
    compare y_scaler to y_pred, just a simple lower/higher, maybe by percentage

'''


#default property values
propertyDefaults = {
            'zpid':'',
            'amount': 0, # property value (rent)
            'valueChange': 0,
                #30-day
            'low': 0,
            'high': 0,
                #valuation range(low to high)
            'percentile': 0,
            'zindexValue': 0,

            'last-updated': '',
            'street': '',
            'zipcode': '',
            'city': '',
            'state': '',
            'latitude': '',
            'longitude': '',

            'FIPScounty': '',
            'useCode': '', 
                 #specifies type of home:
                 #duplex, triplex, condo, mobile, timeshare, etc
            'taxAssessmentYear': '',
                #year of most recent tax assessment
            'taxAssessment': 0,

            'yearBuilt': '',
            'lotSizeSqFt': 0,
            'finishedSqFt': 0,
            'bathrooms': 0,
            'bedrooms': 0,
            'lastSoldDate': '',
            'lastSoldPrice': 0,
            
            
        }
'''
one-hot encodings:
    useCode
    zipcode (first three digits)

'''


# %% API Call

zill = House(propertyDefaults)

'''
search function:
    address field 1: number + street
            field 2: city           (must be in LA county)
            field 3: state, ZIP     (only in CA for now)
'''

#preapproved list of cities in LA county available for model
la_county_cities = []

address = { 'street' : '7101 Colbath Ave',
            'city' : 'Van Nuys CA'}


zill.update(address)

zill.deep_search(key)
zill.get_zestimate(key)



# %% Add Crime, Schools

crime_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/crime_density_rh_gridsize_1.csv'
crime = pd.read_csv(crime_file)

zill.get_crime_density(crime)

schools_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/schools/schools.shp'
districts_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/school_districts_bounds/School_District_Boundaries.shp'

zill.add_schools(schools_file, districts_file)


# %% Transform House to Scaler() from Model


_x_scaler = joblib.load('ml/data/x_scaler.gz')
_y_rent_scaler = joblib.load('ml/data/y_rent_scaler.gz')
_y_zest_scaler = joblib.load('ml/data/y_zest_scaler.gz')

_x_cols = joblib.load('ml/data/x_cols.pkl')
_y_rent_col = joblib.load('ml/data/rent_col.pkl')
_y_zest_col = joblib.load('ml/data/zest_col.pkl')

_predictor_cols = joblib.load('ml/data/predictor_cols.pkl')

zill.transform(_x_scaler, _y_rent_scaler, _x_cols, _y_rent_col, _predictor_cols)




# %% Convert to Tensor

_x_tensor, _y_tensor = zill.get_tensor()

# %%

model.eval()

y_pred = model(_x_tensor)

val_loss = criterion(y_pred, y_tensor)


# %% Loading the model
D_in, D_out = x_tensor.shape[1], y_tensor.shape[1]

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
PATH = 'state_dict_model.pt'

#create device object for cuda operations
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#load model
test_model = Net(D_in, D_out, L1, L2, L3, L4)

test_model.load_state_dict(torch.load(PATH))


test_model.eval()

y_pred = test_model(x_tensor)

