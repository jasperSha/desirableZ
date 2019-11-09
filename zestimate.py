import requests
import xml.etree.ElementTree as ET

key = 'X1-ZWz1hgrt0pjaiz_1brbp'
zpid = '21212400'

parameters = {
    'zws-id':key,
    'zpid':zpid
}

class PropertyZest:
        def __init__(self, d):
            self.__dict__ = d

        def outputValues(self):
            print(self.__dict__)



retrievalList = (
    'amount',
    'valueChange',
    'low',
    'high',
    'percentile',
    'zindexValue',
    'zipcode-id',
    'city-id',
    'county-id',
    'state-id'
    )

retrievalCategories = dict.fromkeys([
    'amount',
    'valueChange',
    'low',
    'high',
    'percentile',
    'zindexValue',
    'zipcode-id',
    'city-id',
    'county-id',
    'state-id'
])

response = requests.get("https://www.zillow.com/webservice/GetZestimate.htm",
                        params = parameters)

root = ET.fromstring(response.content)
for category in retrievalList:
    for child in root.iter('%s'%category):
        retrievalCategories['%s'%category] = child.text

zillowObject = PropertyZest(retrievalCategories)

zillowObject.outputValues()
