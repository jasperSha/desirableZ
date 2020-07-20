import os
from dotenv import load_dotenv
load_dotenv()

""" 
KEYS:
    ZILLOW_API_KEY
    GREATSCHOOLS_API_KEY
    SOCRATA_CRIME_DATA_KEY


"""
key = os.getenv('ZILLOW_API_KEY')


from zillowObject import zillowObject
from zestimate import get_zestimate
from deepsearchresults import deep_search
import postgrestaccess
from zipcode_parse import parse_gpkg as gpkg
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
import requests



#default property values
propertyDefaults = {
            'zpid':'',
            'amount': 0, # property value (rent)
            'valueChange': 0,
                #30-day
            'low': 0,
            'high': 0,
                #valuation range(low to high)
            'percentile': 0,
            'zindexValue': 0,

            'last-updated': '',
            'street': '',
            'zipcode': '',
            'city': '',
            'state': '',
            'latitude': '',
            'longitude': '',

            'FIPScounty': '',
            'useCode': '', 
                 #specifies type of home:
                 #duplex, triplex, condo, mobile, timeshare, etc
            'taxAssessmentYear': '',
                #year of most recent tax assessment
            'taxAssessment': 0,

            'yearBuilt': '',
            'lotSizeSqFt': 0,
            'finishedSqFt': 0,
            'bathrooms': 0,
            'bedrooms': 0,
            'lastSoldDate': '',
            'lastSoldPrice': 0
        }




def run_raw_address(citystatezip, address): #wrap into a function elsewhere?
    try:
        # zillowProperty = zillowObject.PropertyZest(propertyDefaults) #init zillowProperty as a zillowObject(init with given dict)
        zillowProperty = propertyDefaults
        #deep search call
        deep_search(key, citystatezip, address, zillowProperty) 

        if (zillowProperty['zpid'] != ''): #making sure property is listed/not null, else continue
            #zestimate call
            get_zestimate(key, zillowProperty['zpid'], zillowProperty) 
            
            #clean formatting
            zillowProperty['zindexValue'] = str(zillowProperty['zindexValue']).replace(',','')
            zillowProperty['lastupdated'] = zillowProperty['last-updated']
            del zillowProperty['last-updated']
            
            return zillowProperty
            #upload property to postgresql db
            # print('recording zillow property: %s'%zillowProperty)
            # postgrestaccess.record_zillowProperty(zillowProperty)
        else:
            print('this address has no zpid, continuing..')
            
            
    except:
        print("address failure somewhere")
        
        
'''
TODO:
    build update function for zillow properties and compiling into shapefile
'''        

def retrieve_zpid():
    os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
    awz = gpd.read_file('zillowdb/zillowaws.shp')
    awz.drop_duplicates(subset=['street', 'city'],inplace=True)

    streets = awz['street'].values.tolist()
    city = awz['city'].values.tolist()
    
    addresses = list(zip(streets, city))
    awz['zpid'] = ''
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ')
    
    for index, house in enumerate(addresses):
        url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'
        parameters = {
            "zws-id": key,
            'address': house[0], #format: number+street+dr dr or drive ok
            'citystatezip': house[1] #format: city+state_abbreviation
        }
        response = requests.get(url, params=parameters)
        root = ET.fromstring(response.content)
        # print("Grabbing deepsearch values now...")
        for element in root.iter('zpid'):
            zpid = element.text
        awz.at[index, 'zpid'] = zpid
    return awz

if __name__=='__main__':
    
        
        
        
        