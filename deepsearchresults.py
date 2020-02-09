import requests
import xml.etree.ElementTree as ET
#takes zillowObject, updates its dict with deep_search specific attributes
def deep_search(key, url, citystatezip, address, zillowObject, deepPropAttr):
    try:
        # retrievalCategories = vars(zillowObject)
        parameters = {
            "zws-id": key,
            'citystatezip':citystatezip, #format: city+state_abbreviation
            'address': address #format: number+street+dr dr or drive ok
        }
    
        response = requests.get(url, params=parameters)
        root = ET.fromstring(response.content)
        # print("Grabbing deepsearch values now...")
        for category in deepPropAttr:
            for child in root.iter('%s' % category):
                zillowObject['%s'%category] = child.text
        
        print('Property Values updated by DeepSearch.')
        return zillowObject
    except requests.exceptions.Timeout:
        print('connection timeout.. possible throttling')
    except requests.exceptions.ConnectionError:
        print('bad connection')
        
