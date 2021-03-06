from collections.abc import MutableMapping
import requests
import xml.etree.ElementTree as ET
import joblib

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from geospatial.to_wkt import to_wkt
from geospatial.schoolimputation import property_school_rating
from ml.kdtree import knearest_balltree


class House(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._values = dict()
        self.update(dict(*args, **kwargs))
    
    def __getitem__(self, key):
      return self._values[key]
  
    def __getattr__(self, attr):
        return self.get(attr)

    def __setitem__(self, key, value):
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]
    
    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __contains__(self, item):
        if item in self._values.keys():
            return True
        return False
    
    def pop(self, k):
        return self._values.pop(k)
    
    def keys(self):
        return self._values.keys()
    
    def items(self):
        return self._values.items()
    
    def values(self):
        return self._values.values()
    
    def _to_df(self):
        return pd.DataFrame([self.values()], columns=self.keys())
        
    def _to_gdf_wkt(self):
        df = self._to_df()
        
        #convert to geodataframe for geospatial calculations
        geometry = [Point(df['longitude'], df['latitude'])]
        df = df.drop(['longitude', 'latitude'], axis=1)
        
        return gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=geometry)        
        
    
    def get_zestimate(self, key):
        try:
            url = 'https://www.zillow.com/webservice/GetZestimate.htm'
            parameters = {
                'zws-id':key,
                'zpid':self._values['zpid'],
                'rentzestimate':True #necessary for rental value
            }
            response = requests.get(url,params = parameters)
            root = ET.fromstring(response.content)
            
            # print("Grabbing zestimate attributes now...")
            _attribs = {}
            for key in self._values.keys():
                for child in root.iter('%s'%key):
                    _attribs[key] = child.text
            
            #delineate between rent and total value
            _attribs['rentzestimate'] = _attribs.pop('amount')
            self._values['zestimate'] = self._values.pop('amount')
            
            self._values.update(_attribs)
            
            self._values['zindexValue'] = str(self._values['zindexValue']).replace(',', '')
            self._values['lastupdated'] = self._values.pop('last-updated')
    
        except Exception as e:
            print(e)
    
    def deep_search(self, key):
        try:
            url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'
            
            parameters = {
                "zws-id": key,
                'citystatezip': self._values['city'], #format: city+state_abbreviation
                'address': self._values['street'] #format: number+street+dr dr or drive ok
            }
        
            response = requests.get(url, params=parameters)
            root = ET.fromstring(response.content)
            
            _attribs = {}
            for k in self._values.keys():
                for child in root.iter('%s'%k):
                    _attribs[child.tag] = child.text
            
            self._values.update(_attribs)
                    
        except requests.exceptions.Timeout:
            print('connection timeout.. possible throttling')
        except requests.exceptions.ConnectionError:
            print('bad connection')
    
    def get_crime_density(self, crime: pd.DataFrame):

        crime_gdf = to_wkt(crime)
        _self_gdf = self._to_gdf_wkt()
        
        with_crime_df = knearest_balltree(_self_gdf, crime_gdf, radius=1.1)
        density = with_crime_df['crime_density'].iloc[0]
        
        crime_attrib = { 'crime_density' : density}
        self._values.update(crime_attrib)
        
    def add_schools(self, schools_file, districts_file):
        
        #first find district of house
        _self_gdf = self._to_gdf_wkt()
        
        schools_df = gpd.read_file(schools_file)
        schools_df.crs = 'EPSG:4326'
        
        districts_df = gpd.read_file(districts_file)
        districts_df.crs = 'EPSG:4326'

        _self_joined_districts_gdf = gpd.sjoin(_self_gdf, districts_df, op='within')
        _cols = _self_gdf.columns.values.tolist()
        _cols.append('DISTRICT')
        _self_joined_districts_gdf = _self_joined_districts_gdf[_cols]
        
        _self_gdf = property_school_rating(_self_joined_districts_gdf, schools_df)
        
        edu_rating = _self_gdf['edu_rating'].iloc[0]
        
        school_attrib = { 'edu_rating' : edu_rating }
        
        self._values.update(school_attrib)

    def transform(self, x_scaler, y_scaler, x_cols, y_col, predictor_cols):
        '''
        apply scaling from trained data set,
        apply log normalization of crime density
        apply one-hot encoding for use code and zipcode
        
        if passing rent column, set member variable _rent = True,
        so that converting to Tensor function will drop the appropriate
        column from dataframe
        
        sets member pandas dataframe
        
        '''
        
        if y_col == ['rentzestimate']:
            self._rent = True
        else:
            self._rent = False
            
        self_df = self._to_df()
        
        #api returns as strings, converting to float for scaling
        str_to_float_cols = ['valueChange', 'low', 'high', 'percentile',
                             'zindexValue', 'zipcode', 'taxAssessment',
                             'lotSizeSqFt', 'finishedSqFt', 'bathrooms',
                             'bedrooms', 'lastSoldPrice', 'zestimate',
                             'rentzestimate']
        
        self_df[str_to_float_cols] = self_df[str_to_float_cols].apply(pd.to_numeric)
        
        
        #apply model scaling
        self_df[x_cols] = x_scaler.transform(self_df[x_cols])
        self_df[y_col] = y_scaler.transform(self_df[y_col])
        
        #apply crime log norm
        self_df['crime_density'] = self_df['crime_density'].apply(lambda x: x + 1)
        self_df['crime_density'] = np.log(self_df['crime_density'])
        
        #reduce zipcode to first 3 digits
        self_df['zipcode'] = self_df['zipcode'].apply(lambda x: x // 100)
        
        #mark one-hot for use code
        usecode = self_df['useCode'].iloc[0]
        
        #zipcode one-hot encoding column names are string repr of floats (I need to fix this in modelbuilder)
        zipcode = str(float(self_df['zipcode'].iloc[0]))
        
        #preserve dependent variables before reindexing with predictor columns
        y_self_df = self_df[y_col]
        x_self_df = self_df.reindex(labels=predictor_cols, axis=1, fill_value=0)
        
        self_df = pd.concat([x_self_df, y_self_df], axis=1, join='inner').reset_index(drop=True)
        
        for col in self_df.columns:
            if (zipcode == col or usecode == col):
                self_df[col] = 1
        
        #hacky way to impute for NaN ValueChange
        if math.isnan(self_df['valueChange']):
            self_df['valueChange'] = (self_df['low'].iloc[0] + self_df['high'].iloc[0]) / 2.0
        
        self._df = self_df
        
    def get_tensor(self):
        '''
        Converts dependent and independent values to tensors.
            - if rent -> dependent represents rentzestimate
                 else -> dependent represents Zestimate
        
        Can only be executed AFTER transform(), both for normalization and
        instantiation of dataframe member.
        Keeps original dataframe.
        '''
        
        y_col = ['rentzestimate'] if self._rent else ['zestimate']
        y = pd.DataFrame(self._df, columns=y_col)
        x = self._df.drop(y_col, axis=1)
        
        x = torch.tensor(x.values, dtype=torch.float)
        y = torch.tensor(y.values, dtype=torch.float)
        
        return x, y
        
        
        
    
