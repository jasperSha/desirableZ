from zillowObject import zillowObject
from zestimate import get_zestimate as ZT

key = 'X1-ZWz1hgrt0pjaiz_1brbp'
zpid = '21212400'

defaults = {
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


if __name__=='__main__':
    zillowProperty = zillowObject.PropertyZest(defaults)
    print(ZT(key, zpid, zillowProperty).returnValues())
