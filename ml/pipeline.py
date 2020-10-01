import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml.kdtree import knearest_balltree

from zestimate import get_zestimate
from deepsearchresults import deep_search
from ml.model.neuralnet import Net

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


# %% API Call
key = os.getenv('ZILLOW_API_KEY')



'''
search function:
    address field 1: number + street
            field 2: city           (must be in LA county)
            field 3: state, ZIP     (only in CA for now)
'''

#preapproved list of cities in LA county available for model
la_county_cities = []


house = {}

street = '7101 Colbath Ave'
city = 'Van Nuys CA'

house = deep_search(key, city, street, house)
house = get_zestimate(key, house[zpid], house)


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



