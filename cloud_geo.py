import os
from dotenv import load_dotenv
import requests
# from googleapiclient.discovery import build

load_dotenv()

google_key = os.getenv('GOOGLE_CLOUD_API_KEY')

url = 'https://maps.googleapis.com/maps/api/geocode/json'

#geocode in latitude/longitude format
lat = '34.1015999'
lon = '-118.3387'


params = {
    'key' : google_key,
    'latlng':'%s, %s'%(lat, lon) 
    }

response = requests.get(url,params=params)
root = response.json()


addresses = []
for i in root.results.formatted_address:
    print(i)
    