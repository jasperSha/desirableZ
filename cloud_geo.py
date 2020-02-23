import os
from dotenv import load_dotenv
import requests
import xml.etree.ElementTree as ET
load_dotenv()
google_key = os.getenv('GOOGLE_CLOUD_API_KEY')

url = 'https://maps.googleapis.com/maps/api/geocode/xml'

""" 4 decimal places in long/latitude for about 11m precision (parcel of land)
    5 decimal places for about 1.1m precision (individual trees)
"""

#geocode in latitude/longitude format
lat = '34.0953'
lon = '-118.2717'


western_lat = '34.05518'


params = {
    'key' : google_key,
    'latlng':'%s %s'%(lat, lon),
    'result_type':'street_address'
    # 'location_type':'RANGE_INTERPOLATED'
    }

response = requests.get(url,params=params)
root = ET.fromstring(response.content)

for child in root.iter():
    print(child.text)
for child in root.iter('status'):
    print(child.text)
for child in root.iter('formatted_address'):
    print(child.text, 'found at (%s, %s)'%(lat, lon))
    
    