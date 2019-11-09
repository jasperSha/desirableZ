import requests
import xml.etree.ElementTree as ET

url = "https://zillow.com/webservice/GetDeepSearchResults.htm"
key = 'X1-ZWz1hgrt0pjaiz_1brbp'
citystatezip = 'LongBeach+CA'
address = '6415+E+Bixby+Hill+Rd'


parameters = {
    "zws-id": key,
    'citystatezip':citystatezip,
    'address': address
}



response = requests.get(url, params=parameters)


root = ET.fromstring(response.content)
print(response.content)
for child in root.iter('*'):
    print(child.tag, child.text)
