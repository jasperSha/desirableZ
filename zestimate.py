import requests
import xml.etree.ElementTree as ET
from zillowObject import zillowObject
#take zillowObject, update with zestimate specific attributes (requires zpid to access)
def get_zestimate(key, zpid, url, zillowObject, zestPropAttr):
    
    retrievalCategories = vars(zillowObject)
    parameters = {
        'zws-id':key,
        'zpid':zpid,
        'rentzestimate':'true'
    }
    
    response = requests.get(url,params = parameters)
    root = ET.fromstring(response.content)
    
    for category in zestPropAttr:
        for child in root.iter('%s'%category):
            retrievalCategories['%s'%category] = child.text

    zillowObject.update(**retrievalCategories)
    
    return zillowObject
