
zillowkey = 'X1-ZWz1hgrt0pjaiz_1brbp'
zpid = '21212400

retrievalCategories = dict.fromkeys([
    'amount',
    'valueChange',
    'low',
    'high',
    'percentile',
    'zindexValue',
    'zipcode-id',
    'city-id',
    'county-id',
    'state-id',

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
])


class PropertyZest:
        def __init__(self, d):
            self.__dict__ = d

        def outputValues(self):
            print(self.__dict__)




if __name__=='__main__':
    
