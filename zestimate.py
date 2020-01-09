import requests
import xml.etree.ElementTree as ET
#take zillowObject, update with zestimate specific attributes 
#(requires zpid to access)
def get_zestimate(key, zpid, url, zillowObject, zestPropAttr):
    
    retrievalCategories = vars(zillowObject)
    parameters = {
        'zws-id':key,
        'zpid':zpid,
        'rentzestimate':True #necessary for rental value
    }
    
    response = requests.get(url,params = parameters)
    root = ET.fromstring(response.content)
    
    for category in zestPropAttr:
        for child in root.iter('%s'%category):
            retrievalCategories['%s'%category] = child.text

    zillowObject.update(**retrievalCategories)
    print('Property Values updated by Zestimate.')
    
    return zillowObject
