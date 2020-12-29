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
    sample = pd.read_csv(path + 'sample_500_zestimate.csv')
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
    cwd = os.getcwd()

    # Load Model Scalers
    _x_scaler = joblib.load('ml/data/x_scaler.gz')
    _y_rent_scaler = joblib.load('ml/data/y_rent_scaler.gz')
    _y_zest_scaler = joblib.load('ml/data/y_zest_scaler.gz')
    
    _x_cols = joblib.load('ml/data/x_cols.pkl')
    _y_rent_col = joblib.load('ml/data/rent_col.pkl')
    _y_zest_col = joblib.load('ml/data/zest_col.pkl')
    
    #predictor cols also determines whether we run crime density or even school check
    _predictor_cols = joblib.load('ml/data/predictor_cols_03.pkl')
    

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
    
    '''
    TODO: some zillow properties don't have zestimate(home value) values, only rental costs. have to skip those ones if so. (or choose to only model based on rental costs)
    SOLUTION: our sample zillow properties will only be single-family HOMES, with home values and NOT rent(ie no apartment complexes)
    '''
    zill.get_zestimate(key)

    # Add Crime    
    if 'crime_density' in _predictor_cols:
        crime_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/crime_density_rh_gridsize_1.csv'
        crime = pd.read_csv(crime_file)
        
        zill.get_crime_density(crime)
    
    # Add Schools
    if 'edu_rating' in _predictor_cols:
        schools_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/schools/schools.shp'
        districts_file = '/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/data/school_districts_bounds/School_District_Boundaries.shp'
        
        zill.add_schools(schools_file, districts_file)
        
    #add proximal locations
    zill.get_prox()
        

    # Normalization
    
    # #using zestimate as output
    zill.transform(_x_scaler, _y_zest_scaler, _x_cols, _y_zest_col, _predictor_cols)
    
    # using rent as output
    # zill.transform(_x_scaler, _y_rent_scaler, _x_cols, _y_rent_col, _predictor_cols)

    # Retrieve Tensors
    _x_tensor, _y_tensor = zill.get_tensor()
    
    # Loading the model
    D_in, D_out = _x_tensor.shape[1], _y_tensor.shape[1]
    L1, L2, L3, L4 = 4000, 5000, 5000, 4000
    criterion = nn.MSELoss()
    
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')
    model = 'state_dict_model_03.pt'
    
    # #init model
    model_prod = Net(D_in, D_out, L1, L2, L3, L4)
    model_prod.load_state_dict(torch.load(model))
    
    
    model_prod.eval()
    
    # Test against Model
    y_pred = model_prod(_x_tensor)
    
    loss = torch.sqrt(criterion(y_pred, _y_tensor))
    
    # print('For the property located at: ', address['street'], ' ', address['city'])
    # print("percentage predicted: ", y_pred/_y_tensor)
    # print('loss: ', loss.item())
    
    actual = _y_tensor.detach().numpy()
    prediction = y_pred.detach().numpy()
    
    y_actual_scaled = _y_zest_scaler.inverse_transform(actual).item()
    y_pred_scaled = _y_zest_scaler.inverse_transform(prediction).item()
    
    # print("Actual value: ", y_actual_scaled, "Predicted value: ", y_pred_scaled)
    
    scaled_loss = y_actual_scaled - y_pred_scaled
    results.append(tuple((y_actual_scaled, y_pred_scaled, scaled_loss)))
    
    os.chdir(cwd)
    return zill

# %%

pd.options.display.max_columns = None
address_list = generate_addresses()

   
results = []
houses = []

# %%


prev_count = 334
for count, x in enumerate(address_list[prev_count:], start=prev_count):
    try:
        zill = rate_house(x, results)
        print(results[count])
        houses.append(zill)
    except KeyError:
        continue



# %%
overshot, undershot = 0, 0
pct_off = []

for result in results:
    if result[2] < 0:
        overshot += 1
        pct_off.append(result[2] / result[0])
        print('Actual price: ', result[0], ' Predicted price: ', result[1], 'Model prediction overshot by: ', result[2])
    if result[2] > 0:
        undershot += 1
        pct_off.append(result[2] / result[0])
        print('Actual price: ', result[0], ' Predicted price: ', result[1], 'Model prediction undershot by: ', result[2])

print('overshot: ', overshot)
print('undershot: ', undershot)

avg_pct_off = np.mean(pct_off)
print(avg_pct_off)



# %% normalizing losses, rescaling from -1 to 1

loss_arr = np.array([result[2] for result in results])
normed = (2.0 * ((loss_arr - min(loss_arr))/(max(loss_arr) - min(loss_arr)))) - 1
np.set_printoptions(suppress=True)

gradients = normed.tolist()

# %% adding evaluations

for zill, res, grad in list(zip(houses, results, gradients)):
    zill.add_evaluation(res[1], grad)


# %%
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
features = [house._values for house in houses]

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/ml/data/')

with open('500_houses.json', 'w') as fout:
    json.dump(features, fout, cls=NpEncoder, indent=4, separators=(',', ': '))
















