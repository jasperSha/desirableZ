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


if __name__=='__main__':
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        