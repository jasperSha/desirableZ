import numpy as np
import geopandas as gpd
import os
import pandas as pd
from dotenv import load_dotenv
import requests
import xml.etree.ElementTree as ET
import pprint

load_dotenv()

yelp_key = os.getenv('YELP_API_KEY')

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
    
def parse_restaurants():
    yelp_URL = 'https://api.yelp.com/v3/businesses/search'
    
    parameters = {
        'location' : 'Los Angeles',
        'sort_by': 'rating'
    }
    headers = {
        'Authorization': 'bearer %s' % yelp_key
        }
    response = requests.get(yelp_URL, params=parameters, headers=headers)
    
    
    pprint.pprint(response.json()['businesses'])
    
def parse_churches():    
    file_dir = r'data/worship/'
        
    df = gpd.read_file(file_dir + 'AllPlacesOfWorship.shp')
    
    df = df['geometry']
    
    return df
    
















