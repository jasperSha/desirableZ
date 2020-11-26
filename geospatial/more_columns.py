#!/usr/bin/env python
from collections.abc import MutableMapping
import numpy as np
import geopandas as gpd
import os
import pandas as pd
from dotenv import load_dotenv
import requests
import xml.etree.ElementTree as ET
import pprint


def parse_public_transpo():
    file_dir = r'data/publictranspo/'
    
    df = gpd.read_file(file_dir + 'Metro_Rail_Lines_Stops.shp')
    df2 =  gpd.read_file(file_dir + 'Metro_Stations.shp')
    
    df = df['geometry']
    
    df2 = df2['geometry']
    
    ptranspo = pd.concat([df, df2])
    
    ptranspo = ptranspo.drop_duplicates()
    return ptranspo



def parse_university_locations():
    file_dir = r'data/university_locations/'
    df = gpd.read_file(file_dir + 'us-colleges-and-universities.shp')
    df = df['geometry']
    
    return df
    
def parse_restaurants(house, radius):
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
    '''
    load_dotenv()
    yelp_key = os.getenv('YELP_API_KEY')
    yelp_URL = 'https://api.yelp.com/v3/businesses/search'

    #longitude = house.longitude
    #latitude = house.latitude
    longitude = '-118.341438'
    latitude = '34.052503'
    radius = 6000
    
    parameters = {
        'longitude' : longitude,
        'latitude' : latitude,
        'sort_by' : 'distance',
        'radius' : radius
    }
    headers = {
        'Authorization': 'bearer %s' % yelp_key
        }
    response = requests.get(yelp_URL, params=parameters, headers=headers)
    sum_found = response.json()['total']
    data = response.json()['businesses']
    
    df = pd.DataFrame.from_records(data)
    return df
    
def parse_churches():    
    file_dir = r'data/worship_locations/'
        
    df = gpd.read_file(file_dir + 'AllPlacesOfWorship.shp')
    
    df = df['geometry']
    
    return df
    



if __name__ == '__main__':
#    df = parse_churches()
    
#    parse_restaurants()
    
    example = ''

    df = parse_restaurants(example, 50)

    df.drop(columns=['id'], axis=1)

    print(df.head(), df.shape, df.columns)











