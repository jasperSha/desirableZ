from zillowObject import zillowObject as zo
from zestimate import get_zestimate as zt
from deepsearchresults import deep_search as ds

key = 'X1-ZWz1hgrt0pjaiz_1brbp' #zillow API key

zestimate_url = 'https://www.zillow.com/webservice/GetZestimate.htm'
deepsearch_url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'

zpid = '20522179' #propertyID found after deepsearch
deep_citystatezip = 'Beverly+Hills+CA' #for deepsearch, only city/state abbreviation
deep_address = '1027+Summit+Dr'#also for deepsearch

#default property values
propertyDefaults = {
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


if __name__=='__main__':
    zillowProperty = zo.PropertyZest(propertyDefaults) #init zillowProperty as a zillowObject(init with given dict)

    #deepsearch done first
    ds(key, deepsearch_url, deep_citystatezip, deep_address, zillowProperty, deepPropAttr) 
    x = vars(zillowProperty)#just checking status of zillowProperty
    
    #getting zestimate next, after grabbing propertyID
    zt(key, x['zpid'], zestimate_url, zillowProperty, zestPropAttr) 
    
    #convert to accessible dict
    x = vars(zillowProperty)
    