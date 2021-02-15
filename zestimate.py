import requests
import xml.etree.ElementTree as ET

def get_zestimate(key, zpid, House):
    '''
    Calls Zillow's get zestimate endpoint.
    @return: House with monthly rental costs
    '''
    try:

        url = 'https://www.zillow.com/webservice/GetZestimate.htm'
        parameters = {
            'zws-id':key,
            'zpid':House.zpid,
            'rentzestimate':True #necessary for rental value
        }
        response = requests.get(url,params = parameters)
        root = ET.fromstring(response.content)
        
        # print("Grabbing zestimate attributes now...")
        attribs = {}
        for k in House.keys():
            for child in root.iter('%s'%k):
                attribs[k] = child.text
        
        #delineate between rent and total value
        attribs['rentzestimate'] = attribs.pop('amount')
        House['zestimate'] = House.pop('amount')
        
        House.update(attribs)
        
        House['zindexValue'] = str(House['zindexValue']).replace(',','')
        House['lastupdated'] = House.pop('last-updated')
        
        return House
    except Exception as e:
        print(e)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

