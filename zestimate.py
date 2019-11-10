import requests
import xml.etree.ElementTree as ET
from zillowObject import zillowObject

def get_zestimate(key, zpid, zillowObject, zestPropAttr):
    
    retrievalCategories = vars(zillowObject)
    
    url = 'https://www.zillow.com/webservice/GetZestimate.htm'
    
    parameters = {
        'zws-id':key,
        'zpid':zpid
    }
    
    response = requests.get(url,params = parameters)
    root = ET.fromstring(response.content)
    
    for category in zestPropAttr:
        for child in root.iter('%s'%category):
            retrievalCategories['%s'%category] = child.text

    zillowObject.update(**retrievalCategories)
    return zillowObject
