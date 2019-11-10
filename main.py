from zillowObject import zillowObject as zo
from zestimate import get_zestimate as zt
from deepsearchresults import deep_search as ds

key = 'X1-ZWz1hgrt0pjaiz_1brbp'
zpid = '21212400'
zestimate_url = 'https://www.zillow.com/webservice/GetZestimate.htm'
deepsearch_url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'
deep_citystatezip = 'LongBeach+CA'
deep_address = '6415+E+Bixby+Hill+Rd'

#default property values
propertyDefaults = {
            'amount': 0,
            'valueChange': 0,
            'low': 0,
            'high': 0,
            'percentile': 0,
            'zindexValue': 0,
            'zipcode-id': '',
            'city-id': '',
            'county-id': '',
            'state-id': '',

            'last-updated': '',
            'street': '',
            'zipcode': '',
            'city': '',
            'state': '',
            'latitude': '',
            'longitude': '',

            'FIPScounty': '',
            'useCode': '',
            'taxAssessmentYear': '',
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
            'zindexValue',
            'zipcode-id',
            'city-id',
            'county-id',
            'state-id'
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
    zillowProperty = zo.PropertyZest(propertyDefaults)
    zt(key, zpid, zestimate_url, zillowProperty, zestPropAttr)
    ds(key, deepsearch_url, deep_citystatezip, deep_address, zillowProperty, deepPropAttr)
