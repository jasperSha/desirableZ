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





#default property values
propertyDefaults = {
            'zpid':'',
            'amount': 0, # property value
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

#property attributes unique to zestimate call
zestPropAttr = (
            'amount',
            'valueChange',
            'low',
            'high',
            'percentile',
            'zindexValue'
        )

#property attributes unique to deepsearch call
deepPropAttr = (
            'zpid',
            
            'last-updated',
            
            'street',
            'zipcode',
            'city',
            'state',
            'latitude',
            'longitude',
            
            'FIPScounty',
            'useCode',
            'taxAssessmentYear',
            'taxAssessment',
            
            'yearBuilt',
            'lotSizeSqFt',
            'finishedSqFt',
            'bathrooms',
            'bedrooms',
            
            
            'lastSoldDate',
            'lastSoldPrice'
        )


def run_raw_address(citystatezip, address): #wrap into a function elsewhere?
    try:
        zillowProperty = zillowObject.PropertyZest(propertyDefaults) #init zillowProperty as a zillowObject(init with given dict)

        #deep search api call
        deep_search(key, deepsearch_url, citystatezip, address, zillowProperty, deepPropAttr) 
        
        print(zillowProperty)
    
        if zillowProperty['zpid']!='': #making sure property is listed/not null, else continue
            zpid = zillowProperty['zpid']
            #zestimate call
            get_zestimate(key, zpid, zestimate_url, zillowProperty, zestPropAttr) 
            
            #setting all empty strings to NULL for database
            for index, value in zillowProperty.items():
                if value == '':
                    zillowProperty[index] = None
            
            print(zillowProperty)
            #upload property to postgresql db
            # print('recording zillow property: %s'%zillowProperty)
            # postgrestaccess.record_zillowProperty(zillowProperty)
        else:
            print('this address has no zpid, continuing..')
            
            
    except:
        print("address failure somewhere")
        


zestimate_url = 'https://www.zillow.com/webservice/GetZestimate.htm'
deepsearch_url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'

# deep_citystatezip = 'Palos Verdes Peninsula CA' #for deepsearch, only city/state abbreviation
# deep_address = '1+Dapplegray+Lane'#also for deepsearch

if __name__=='__main__':
    # zillowProperty = zillowObject.PropertyZest(propertyDefaults) #init zillowProperty as a zillowObject(init with given dict)

    # #deepsearch done first
    # deep_search(key, deepsearch_url, deep_citystatezip, deep_address, zillowProperty, deepPropAttr) 
    
    # print(zillowProperty)
    
    # addresses = postgrestaccess.pull_crime_data()
    
    #testing api endpoint here
    run_raw_address('Beverly Hills CA', '815 N Whittier Dr')
    
    # count = 0
    # for address in addresses:
    #     citystatezip = address[0]
    #     deep_address = address[1]
    #     count+=1
    #     print('running address number %s'%count)
    #     run_raw_address(citystatezip, deep_address)
    
    """ 
    ZILLOW THROTTLED ON 2/8/20.
    NEXT UPDATE FROM ROW 6675 ONWARDS.
    
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        