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


x_scaler = joblib.load('ml/data/x_scaler.gz')
y_scaler = joblib.load('ml/data/y_scaler.gz')
x_cols = joblib.load('ml/data/x_cols.pkl')
y_cols = joblib.load('ml/data/y_cols.pkl')
predictor_cols = joblib.load('ml/data/predictor_cols.pkl')

zill_df = zill.transform(x_scaler, y_scaler, x_cols, y_cols, predictor_cols)


# %% Convert to Tensor

#finally call model(x) -> compare with actual value

y_col = ['zestimate']
y = pd.DataFrame(zill_df, columns=y_col)
x = zill_df.drop(['rentzestimate', 'zestimate'], axis=1)



# %% Loading the model

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/model/')
PATH = 'state_dict_model.pt'

#load model
model = Net(D_in, D_out)
model.load_state_dict(torch.load(PATH))
model.eval()



