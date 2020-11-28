#!/usr/bin/env python
from collections.abc import MutableMapping
import os
from dotenv import load_dotenv

import requests
import xml.etree.ElementTree as ET
import pprint

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads


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
    df = gpd.read_file(file_dir + 'us-colleges-and-universities.shp')
    
    colleges_gdf = gpd.GeoDataFrame(df['geometry'], crs='EPSG:4326', geometry='geometry')
    return colleges_gdf
    
def prox_restaurants(house, radius, category, limit=50):
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

    #longitude = house.longitude
    #latitude = house.latitude
    '''
    TESTING
    '''
    longitude = '-118.341438'
    latitude = '34.052503'
    radius = 40000
    categories = ['food', 'nightlife', 'publicservicesgovt', 'hotelstravel', 'localflavor', 'arts', 'religiousorgs']
    '''
    TESTING
    '''
    
    parameters = {
        'longitude' : longitude,
        'latitude' : latitude,
        'sort_by' : 'distance',
        'categories' : categories,
        'radius' : radius,
        'limit' : limit
    }
    headers = {
        'Authorization': 'bearer %s' % yelp_key
        }
    response = requests.get(yelp_URL, params=parameters, headers=headers)
    sum_found = response.json()['total']
    data = response.json()['businesses']
    
    df = pd.DataFrame.from_records(data)
    df = df.drop(columns=['id', 'alias', 'categories', 'image_url', 'is_closed', 'url', 'display_phone', 'phone', 'location', 'transactions'], axis=1)
#    df = df[df['review_count'] >= 25]
    print(df.head(), df.shape)
    print(df.columns)
    return df

def prox_public(house, radius=8000, category='publicservicesgovt', limit=50):
    '''
    Finds nearest public schools
    '''
    load_dotenv()
    yelp_key = os.getenv('YELP_API_KEY')
    yelp_URL = 'https://api.yelp.com/v3/businesses/search'

    #longitude = house.longitude
    #latitude = house.latitude
    '''
    TESTING
    '''
    longitude = '-118.341438'
    latitude = '34.052503'
    radius = 40000
    categories = category
    '''
    TESTING
    '''
    
    parameters = {
        'longitude' : longitude,
        'latitude' : latitude,
        'sort_by' : 'distance',
        'categories' : categories,
        'radius' : radius,
        'limit' : limit
    }
    headers = {
        'Authorization': 'bearer %s' % yelp_key
        }
    response = requests.get(yelp_URL, params=parameters, headers=headers)
    sum_found = response.json()['total']
    data = response.json()['businesses']
    
    df = pd.DataFrame.from_records(data)
    df = df.drop(columns=['id', 'alias', 'categories', 'image_url', 'is_closed', 'url', 'display_phone', 'phone', 'location', 'transactions'], axis=1)
#    df = df[df['review_count'] >= 25]
    print(df.head(), df.shape)
    print(df.columns)
    
def parse_churches():    
    file_dir = r'data/worship_locations/'
        
    df = gpd.read_file(file_dir + 'AllPlacesOfWorship.shp')
    
    df = df['geometry']
    
    return df
    



if __name__ == '__main__':



    df = prox_colleges()
    print(df.head())









