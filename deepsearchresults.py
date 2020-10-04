import requests
import xml.etree.ElementTree as ET
#takes zillowObject, updates its dict with deep_search specific attributes
# def deep_search(key, citystatezip, address, zillowObject):
#     try:
#         #property attributes unique to deepsearch call
#         deepPropAttr = (
#             'zpid',
            
#             'last-updated',
            
#             'street',
#             'zipcode',
#             'city',
#             'state',
#             'latitude',
#             'longitude',
            
#             'FIPScounty',
#             'useCode',
#             'taxAssessmentYear',
#             'taxAssessment',
            
#             'yearBuilt',
#             'lotSizeSqFt',
#             'finishedSqFt',
#             'bathrooms',
#             'bedrooms',
            
            
#             'lastSoldDate',
#             'lastSoldPrice'
#         )

        
#         url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'
#         parameters = {
#             "zws-id": key,
#             'citystatezip':citystatezip, #format: city+state_abbreviation
#             'address': address #format: number+street+dr dr or drive ok
#         }
    
#         response = requests.get(url, params=parameters)
#         root = ET.fromstring(response.content)
#         # print("Grabbing deepsearch values now...")
#         for category in deepPropAttr:
#             for child in root.iter('%s' % category):
#                 zillowObject['%s'%category] = child.text
        
#         # print('Property Values updated by DeepSearch.')
#         return zillowObject
#     except requests.exceptions.Timeout:
#         print('connection timeout.. possible throttling')
#     except requests.exceptions.ConnectionError:
#         print('bad connection')
        

def deep_search(key, House):
    try:

        url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'
        
        parameters = {
            "zws-id": key,
            'citystatezip': House.city, #format: city+state_abbreviation
            'address': House.street #format: number+street+dr dr or drive ok
        }
    
        response = requests.get(url, params=parameters)
        root = ET.fromstring(response.content)
        
        attribs = {}
        for k in House.keys():
            for child in root.iter('%s'%k):
                attribs[child.tag] = child.text
        House.update(attribs)
                
        # print('Property Values updated by DeepSearch.')
        return House
    except requests.exceptions.Timeout:
        print('connection timeout.. possible throttling')
    except requests.exceptions.ConnectionError:
        print('bad connection')
        





