import requests
import xml.etree.ElementTree as ET
#take zillowObject, update with zestimate specific attributes 
#(requires zpid to access)
def get_zestimate(key, zpid, zillowObject):
    try:
        #property attributes unique to zestimate call
        zestPropAttr = (
                'amount',
                'valueChange',
                'low',
                'high',
                'percentile',
                'zindexValue'
                )
        '''
        first iteration grabbing the property ownership value
        '''
        
        url = 'https://www.zillow.com/webservice/GetZestimate.htm'
        parameters = {
            'zws-id':key,
            'zpid':zpid,
            'rentzestimate':False #necessary for rental value
        }
        response = requests.get(url,params = parameters)
        root = ET.fromstring(response.content)
        
        # print("Grabbing zestimate attributes now...")
        for category in zestPropAttr:
            for child in root.iter('%s'%category):
                zillowObject['%s'%category] = child.text
        
        zillowObject['zestimate'] = zillowObject['amount']
        
        del zillowObject['amount']
      
        '''
        now adding property rental value
        '''
        parameters = {
            'zws-id':key,
            'zpid':zpid,
            'rentzestimate':True #necessary for rental value
        }
        response = requests.get(url,params = parameters)
        root = ET.fromstring(response.content)
        
        for child in root.iter('amount'):
            zillowObject['rentzestimate'] = child.text
    
        return zillowObject
    except Exception as e:
        print(e)