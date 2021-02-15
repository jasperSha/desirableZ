#!/usr/bin/env python
from collections.abc import MutableMapping
import os
from dotenv import load_dotenv
import pickle
import re

import requests
import xml.etree.ElementTree as ET
import pprint

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads

from scipy.spatial import cKDTree
from geospatial.haversine import haversine


def prox_transpo():
    file_dir = r'data/publictranspo/'
    
    df = gpd.read_file(file_dir + 'Metro_Rail_Lines_Stops.shp')
    df2 =  gpd.read_file(file_dir + 'Metro_Stations.shp')

    
    df = df['geometry']
    df2 = df2['geometry']
    
    ptranspo = pd.DataFrame(pd.concat([df, df2]), columns=['geometry'])
    ptranspo = ptranspo.drop_duplicates()
    ptranspo_gdf = gpd.GeoDataFrame(ptranspo, crs='EPSG:4326', geometry='geometry')
    
    return ptranspo_gdf


def prox_colleges():
    file_dir = r'data/university_locations/'
    gdf = gpd.read_file(file_dir + 'us-colleges-and-universities.shp')
    
    cols = ['geometry']
    colleges_gdf = gdf[cols]
    return colleges_gdf


    
def prox_restaurants(zipcode, radius=40000, limit=50):
    '''
    @param: house with longitude/latitude location to send to api request, plus search radius, aiming to sort by distance from long/lat
    @params:
        "location" : geographical area ie city
        "latitude" : 
        "longitude" : 
        "radius" : int -> radius in meters
        "categories" : categories to filter: see yelp for extended list
        "limit" : int -> limit number of returns, maximum 50
        "sort_by" : suggestion to sort by: best_match(default), rating, review_count, or distance
        "price" : filter by pricing levels: 1 = $, 2 = $$, 3 = $$$, 4 = $$$ can also be comma delimited, ie "1, 2, 3" for inclusive or on 1, 2, 3
    @return: summation of ratings within radius, divided by summation of the cost? maybe just average ratings, average cost, add both as columns

    NOTE:
        categories can be specified for restaurants, but also shopping centers, real estate offices, public services, etc.
        categories = ['food', 'publicservicesgovt', 'active'(outdoors activities), 'arts', 'auto', 'beautysvc', 'education','financialservices', 'health', 'homeservices', 'hotelstravel'(includes transportation), 'localflavor', 'localservices', 'nightlife', 'massmedia', 'pets', 'professional'(services), 'realestate', 'religiousorgs','shopping']
    '''
    load_dotenv()
    yelp_key = os.getenv('YELP_API_KEY')
    yelp_URL = 'https://api.yelp.com/v3/businesses/search'

    category = 'restaurants'
    
    parameters = {
        'location' : zipcode,
        'sort_by' : 'distance',
        'categories' : category,
        'radius' : radius,
        'limit' : limit
    }
    headers = {
        'Authorization': 'bearer %s' % yelp_key
        }
    response = requests.get(yelp_URL, params=parameters, headers=headers)
    try:
        total = response.json()['total']
        data = response.json()['businesses']
    except:
        print(response.text)
        return
    
    df = pd.DataFrame.from_records(data)   
    df = df.loc[df['review_count'] >= 25]
    
    df = df.reset_index(drop=True)
    
    longitudes = pd.Series([coordinates['longitude'] for coordinates in df['coordinates']], dtype='float64')
    latitudes = pd.Series([coordinates['latitude'] for coordinates in df['coordinates']], dtype='float64')

    df['longitude'] = longitudes
    df['latitude'] = latitudes
    
    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
    gdf['geometry'] = gdf.geometry
    cols = ['name', 'review_count', 'rating', 'price', 'geometry']
    
    gdf = gdf[cols]
    

    return gdf, data, total

def zipcode_restaurants():
    '''
    Run zipcodes through yelp API, get 50(yelp's limit) restaurants and their ratings/price/review_count
    Write to shapefile                                                                                        
    '''
    cd = os.getcwd()
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/')
    zipcodes = pd.read_csv('zipcodelocations.csv', sep=';')
    
    zipcodes = zipcodes.loc[zipcodes['City']=='Los Angeles']

    cols = ['Zip']
    zipcodes = zipcodes[cols]
    zips = zipcodes['Zip'].to_list()
    
    frames = []
    failed_zipcodes = []
    for idx, locale in enumerate(zips[100:], start=100):
        try:
            # if idx == 100:
            #     break
            gdf = prox_restaurants(locale)
            gdf['zipcode'] = locale
            frames.append(gdf)
        except:
            print(locale)
            failed_zipcodes.append(locale)

    full_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

    os.chdir(cd)

    return idx, full_gdf, failed_zipcodes
    
    

    
def prox_worship():    
    file_dir = r'data/worship_locations/'
        
    gdf = gpd.read_file(file_dir + 'AllPlacesOfWorship.shp')
    gdf = gdf.loc[gdf['CITY_2']=='LOS ANGELES']
    gdf['geometry'] = gpd.points_from_xy(gdf['X'], gdf['Y'])
    gdf.crs = 'EPSG:4326'
    
    cols = ['geometry']
    gdf = gdf[cols]

    
    
    return gdf
    

def proximal_locations(gdf, flag):
    '''
    flag == 'public':
    Defining easy access public transportaion as being within 5 miles of a house
    First searching k=3 nearest neighbors, then dropping those >5 miles away
    Appending count of nearby transportaion centers to house dataframe
    
    flag == 'schools':
    Also finds proximity to public university education, k=3. Primarily composed of
    technical/trade schools, secondary education, whereas the school imputation/ratings
    from GreatSchools is for elementary -> high school ratings, for children.
    
    '''
    cwd = os.getcwd()
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial/')
    if flag == 'transpo':
        prox_gdf = prox_transpo()
    elif flag == 'colleges':
        prox_gdf = prox_colleges()
    elif flag == 'worship':
        prox_gdf = prox_worship()
    os.chdir(cwd)
    
    houses = gdf
    
    #first reset indices so they line up
    houses.reset_index(drop=True, inplace=True)
    prox_gdf.reset_index(drop=True, inplace=True)
    
    origins = np.array(list(houses.geometry.apply(lambda x: (x.x, x.y))))
    neighbors = np.array(list(prox_gdf.geometry.apply(lambda x: (x.x, x.y))))
    
    btree = cKDTree(neighbors)

    _, idx = btree.query(origins, 3)
    
    origins_idx = np.column_stack((origins, idx))
    
    haversine_distances = []
    for array in origins_idx:
        houses_lon, houses_lat = array[0], array[1]
        
        prox_pts = neighbors[array[2:5].astype(int)]
        public_lon, public_lat = prox_pts[:,0], prox_pts[:,1]
        
        distance = haversine(houses_lon, houses_lat, public_lon, public_lat)
        haversine_distances.append(distance)
    haversine_distances = np.stack(haversine_distances, axis=0)

    ans = np.zeros(len(haversine_distances))
    ans = haversine_distances.mean(axis=1)

    prox_series = pd.Series(ans)
    if flag == 'transpo':
        gdf['transpo_prox'] = prox_series
    elif flag == 'colleges':
        gdf['colleges_prox'] = prox_series
    elif flag == 'worship':
        gdf['worship_prox'] = prox_series
    return gdf


if __name__ == '__main__':
    













