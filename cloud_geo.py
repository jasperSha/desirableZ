import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
load_dotenv()
google_key = os.getenv('GOOGLE_CLOUD_API_KEY')

url = 'https://maps.googleapis.com/maps/api/geocode/json'


def geocode_crime(address):
    params = {
        'key': google_key,
        'address': address
        }
    response = requests.get(url, params=params)
    json_data = response.json()
    if json_data['status'] == 'ZERO_RESULTS':
        return ''
    else:
        long_lat = json_data['results'][0]['geometry']['location']
        return 'POINT(' + str(long_lat['lng']) + ' ' + str(long_lat['lat']) + ')'
    




