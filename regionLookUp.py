import requests
import xml.etree.ElementTree as ET

import os
from dotenv import load_dotenv
load_dotenv()

""" 
KEYS:
    ZILLOW_API_KEY
    GREATSCHOOLS_API_KEY
    SOCRATA_CRIME_DATA_KEY


"""
key = os.getenv('ZILLOW_API_KEY')
state = 'CA'
county = 'county'
url = 'https://www.zillow.com/webservice/GetRegionChildren.htm'

parameters = {
    'zws-id': key,
    'state': state,
    'childtype': 'zipcode'
    }

response = requests.get(url, params=parameters)
root = ET.fromstring(response.content)

for child in root.iter('90720'):
    print(child.tag, child.text)
