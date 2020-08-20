import pandas as pd
import numpy as np
import os
from cloud_geo import geocode_crime
from shapely.geometry import Point
import geopandas as gpd

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
    
    for year_file in range(2013, 2019):
        
        year = pd.read_csv('year%s.csv'%year_file)
        
        
        
        year = year[~year['CATEGORY'].str.contains('|'.join(dont_cares), na=False)]
        year = year[~year['STREET'].str.contains('|'.join(dont_cares), na=False)]
        
        year['STREET'].replace('', np.nan, inplace=True)
        year['CATEGORY'].replace('', np.nan, inplace=True)
        year.dropna(subset = ['STREET'], inplace=True) 
        year.dropna(subset = ['CATEGORY'], inplace=True) 

        year = year[year['CITY'].str.contains('|'.join(needed_cities), na=False)]
        
        #geocode addresses
        year['long_lat'] = np.vectorize(geocode_crime)(year['STREET'])
        
        year.to_csv(r'/Users/Jasper/Documents/HousingMap/lasd_crime_stats/year%sgeocoded.csv'%year_file, index= False)
    

def lnglatsplit(row):
    if (isinstance(row['long_lat'], float)):
        return ''
    num = row['long_lat'].split(', ')
    lat = num[1]
    long = num[0]
    long = long[1:]
    lat = lat[:-1]
    row['lon'] = float(long.strip())
    row['lat'] = float(lat.strip())
    return row

os.chdir('/Users/Jasper/Documents/HousingMap/lasd_crime_stats/')

for year_file in range(2006, 2019):
    
    year = pd.read_csv('year%sgeocoded.csv'%year_file)
    
    year = year.apply(lnglatsplit, axis=1)
    year = year.drop('long_lat', axis=1)
    
    
    year['lon'] = pd.to_numeric(year['lon'], errors='coerce')
    year['lat'] = pd.to_numeric(year['lat'], errors='coerce')
    geometry = [Point(xy) for xy in zip(year['lon'], year['lat'])]
    
    crs = {'init':'epsg:4326'}
    gdf = gpd.GeoDataFrame(year, crs=crs, geometry=geometry)
    gdf = gdf.drop('lat', axis=1)
    gdf = gdf.drop('lon', axis=1)
    gdf.to_file('year%s.gpkg'%year_file, layer='crime', driver='GPKG')

# year2005.to_csv(r'/Users/Jasper/Documents/HousingMap/lasd_crime_stats/year2005geocodedagain.csv', index= False)



# for year_file in range(2005, 2019):
    
#     year = pd.read_csv('year%sgeocoded.csv'%year_file)
    
#     year['long_lat'] = 
#     year['lon']
    
    # #lon/lat to POINT format
    # df['longitude_latitude'] = df['longitude_latitude'].apply(lambda x: to_shape(x).to_wkt())
    # df['longitude_latitude'] = df['longitude_latitude'].apply(wkt.loads)
    # df['date_occ'] = df['date_occ'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # gdf_crime = geopandas.GeoDataFrame(df, geometry='longitude_latitude') 
    # gdf_crime.crs = 'EPSG:4326'
    # return gdf_crime
    # # gdf_schools.to_file("%s_crime.gpkg"%year, layer='crime', driver="GPKG")








        