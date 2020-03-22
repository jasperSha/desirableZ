import pandas as pd
import numpy as np
import os
from cloud_geo import geocode_crime

def geocode_crime_data():
    os.chdir('/Users/Jasper/Documents/HousingMap/lasd_crime_stats/')
    dont_cares = [
                  'BOATING', 'NON-CRIMINAL', 'ACCIDENTS', 
                  'WARRANTS', 'DIVISION', 'FORGERY'
                  ]

    needed_cities = [
                     'PASADENA', 'EL MONTE', 'BURBANK', 
                     'SAN GABRIEL', 'GLENDALE', 
                     'MONTEREY PARK','HUNTINGTON PARK'
                     ]
    
    for year_file in range(2006, 2019):
        
        year = pd.read_csv('year%s.csv'%year_file)
        
        year = year[~year['CATEGORY'].str.contains('|'.join(dont_cares))]
        year = year[~year['STREET'].str.contains('|'.join(dont_cares), na=False)]
        year['STREET'].replace('', np.nan, inplace=True)
        year.dropna(subset = ['STREET'], inplace=True) 
        year = year[year['CITY'].str.contains('|'.join(needed_cities), na=False)]
        
        #geocode addresses
        year['long_lat'] = np.vectorize(geocode_crime)(year['STREET'])
        
        year.to_csv(r'/Users/Jasper/Documents/HousingMap/lasd_crime_stats/year%sgeocoded.csv'%year_file, index= False)
    



geocode_crime_data()













        