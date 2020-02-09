import requests
import xml.etree.ElementTree as ET
#take zillowObject, update with zestimate specific attributes 
#(requires zpid to access)
def get_zestimate(key, zpid, url, zillowObject, zestPropAttr):
    
    parameters = {
        'zws-id':key,
        'zpid':zpid,
        'rentzestimate':True #necessary for rental value
    }
    
    response = requests.get(url,params = parameters)
    root = ET.fromstring(response.content)
    
    print("Grabbing zestimate attributes now...")
    for category in zestPropAttr:
        for child in root.iter('%s'%category):
            zillowObject['%s'%category] = child.text
    
    #clean up zindexValue because zillow puts commas in this number but not others for some reason
    zillowObject['zindexValue'] = zillowObject['zindexValue'].replace(',','')

    # print('Property Values updated by Zestimate.')
    
    return zillowObject
