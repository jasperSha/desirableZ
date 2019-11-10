import requests
import xml.etree.ElementTree as ET
from zillowObject import zillowObject

def get_zestimate(key, zpid, zillowObject):

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
    
    url = 'https://www.zillow.com/webservice/GetZestimate.htm'
    parameters = {
        'zws-id':key,
        'zpid':zpid
    }

    retrievalCategories = zillowObject.returnValues()
    response = requests.get(url,params = parameters)

    root = ET.fromstring(response.content)
    for category in retrievalList:
        for child in root.iter('%s'%category):
            retrievalCategories['%s'%category] = child.text

    zillowObject.setValues(retrievalCategories)
    return zillowObject
