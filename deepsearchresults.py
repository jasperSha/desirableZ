import requests
import xml.etree.ElementTree as ET
  

def deep_search(key, House):
    '''
    Calls Zillow's deep search api.
    @return: House with zpid and zestimate value.
    '''
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
        





