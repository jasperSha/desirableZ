from zillowObject import zillowObject
from zestimate import get_zestimate
from deepsearchresults import deep_search
from postgrestaccess import record_zillowProperty

key = 'X1-ZWz1hgrt0pjaiz_1brbp' #zillow API key

zestimate_url = 'https://www.zillow.com/webservice/GetZestimate.htm'
deepsearch_url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'

deep_citystatezip = 'Palos Verdes Peninsula CA' #for deepsearch, only city/state abbreviation
deep_address = '1+Dapplegray+Lane'#also for deepsearch

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


if __name__=='__main__':
    zillowProperty = zillowObject.PropertyZest(propertyDefaults) #init zillowProperty as a zillowObject(init with given dict)

    #deepsearch done first
    deep_search(key, deepsearch_url, deep_citystatezip, deep_address, zillowProperty, deepPropAttr) 
    
    
    x = vars(zillowProperty)
    
    if x['zpid']!='': #making sure property is listed
        zpid = x['zpid']
        get_zestimate(key, zpid, zestimate_url, zillowProperty, zestPropAttr) 
        
        x = vars(zillowProperty)
        
        #upload property to postgresql db
        # record_zillowProperty(x)
    
    #throwing this into a loop later
        
        
        