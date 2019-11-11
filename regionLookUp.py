import requests
import xml.etree.ElementTree as ET

key = 'X1-ZWz1hgrt0pjaiz_1brbp'
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
