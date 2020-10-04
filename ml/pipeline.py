import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml.kdtree import knearest_balltree

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

street = '7101 Colbath Ave'
city = 'Van Nuys CA'

zill.street = street
zill.city = city

zill = deep_search(key, zill)

zill = get_zestimate(key, zill.zpid, zill)


# %% Append Schools and Crime data

'''
Right now, hardcoding D_in, D_out and predictor_cols
'''

D_in, D_out = 40, 1
predictor_cols = ['bathrooms', 'bedrooms', 'finishedSqFt', 'high', 'lastSoldPrice',
       'lotSizeSqFt', 'low', 'taxAssessment', 'valueChange', 'zindexValue',
       'edu_rating', 'crime_density', 'Cooperative', 'Duplex', 'Miscellaneous',
       'Mobile', 'MultiFamily2To4', 'MultiFamily5Plus', 'Quadruplex',
       'SingleFamily', 'Townhouse', 'Triplex', '901.0', '902.0', '903.0',
       '904.0', '905.0', '906.0', '907.0', '908.0', '910.0', '911.0', '912.0',
       '913.0', '914.0', '915.0', '916.0', '917.0', '918.0', '935.0']




#scale test input
#x_scaler.transform(input.x)
#y_scaler.transform(input.y)
    



#order columns using variable: predictor_cols



#finally call model(x) -> compare with actual value






# %% Loading the model

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/model/')
PATH = 'state_dict_model.pt'

#load model
model = Net(D_in, D_out)
model.load_state_dict(torch.load(PATH))
model.eval()



