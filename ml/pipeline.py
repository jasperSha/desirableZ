import os
import joblib
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from deepsearchresults import deep_search
from ml.model.neuralnet import Net
from zillowObject.zillowObject import House

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ')
key = os.getenv('ZILLOW_API_KEY')

# %% Retrieve Sample Houses
def generate_addresses():
    path = '/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/'
    sample = pd.read_csv(path + 'random_sample_50_zillow.csv')
    streets = sample['street'].values.tolist()
    cities = sample['city'].values.tolist()
    states = sample['state'].values.tolist()
    
    citystate = ['%s %s' % x for x in zip(cities, states)]
    
    zillows = []
    for first, second, third in list(zip(streets, cities, states)):
        x = dict()
        x['street'] = first
        x['city'] = second + ' ' + third
        zillows.append(x)
    return zillows


        

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


# %% API Call



def rate_house(address, results):
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ')

    zill = House(propertyDefaults)

    '''
    search function:
        address field 1: number + street
                field 2: city           (must be in LA county)
                field 3: state, ZIP     (only in CA for now)
    '''
        
    #preapproved list of cities in LA county available for model
    la_county_cities = []
    
    zill.update(address)
    
    zill.deep_search(key)
    zill.get_zestimate(key)

    # Add Crime, Schools
    crime_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/crime_density_rh_gridsize_1.csv'
    crime = pd.read_csv(crime_file)
    
    zill.get_crime_density(crime)
    
    schools_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/schools/schools.shp'
    districts_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/school_districts_bounds/School_District_Boundaries.shp'
    
    zill.add_schools(schools_file, districts_file)
    
    # Load Model Scalers
    _x_scaler = joblib.load('ml/data/x_scaler.gz')
    _y_rent_scaler = joblib.load('ml/data/y_rent_scaler.gz')
    _y_zest_scaler = joblib.load('ml/data/y_zest_scaler.gz')
    
    _x_cols = joblib.load('ml/data/x_cols.pkl')
    _y_rent_col = joblib.load('ml/data/rent_col.pkl')
    _y_zest_col = joblib.load('ml/data/zest_col.pkl')
    
    _predictor_cols = joblib.load('ml/data/predictor_cols.pkl')

    # Normalization
    #using zestimate as output
    zill.transform(_x_scaler, _y_zest_scaler, _x_cols, _y_zest_col, _predictor_cols)
    
    #using rent as output
    # zill.transform(_x_scaler, _y_rent_scaler, _x_cols, _y_rent_col, _predictor_cols)

    # Retrieve Tensors
    _x_tensor, _y_tensor = zill.get_tensor()
    
    # Loading the model
    D_in, D_out = _x_tensor.shape[1], _y_tensor.shape[1]
    L1, L2, L3, L4 = 2000, 2000, 2000, 2000
    criterion = nn.MSELoss()
    
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
    FILE = 'state_dict_model.pt'
    
    #init model
    test_model = Net(D_in, D_out, L1, L2, L3, L4)
    test_model.load_state_dict(torch.load(FILE))
    
    
    test_model.eval()
    
    # Test against Model
    y_pred = test_model(_x_tensor)
    
    loss = torch.sqrt(criterion(y_pred, _y_tensor))
    
    print('For the property located at: ', address['street'], ' ', address['city'])
    print("percentage predicted: ", y_pred/_y_tensor)
    print('loss: ', loss.item())
    
    actual = _y_tensor.detach().numpy()
    prediction = y_pred.detach().numpy()
    
    y_actual_scaled = _y_zest_scaler.inverse_transform(actual).item()
    y_pred_scaled = _y_zest_scaler.inverse_transform(prediction).item()
    
    print("Actual value: ", y_actual_scaled, "Predicted value: ", y_pred_scaled)
    results.append(tuple((y_actual_scaled, y_pred_scaled, loss.item())))
    return results


zillows = generate_addresses()
results = []


for house in zillows:
    rate_house(house, results)
    

















