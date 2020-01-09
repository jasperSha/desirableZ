import psycopg2
from psycopg2 import sql
#pipeline set up for funneling addresses into postgres database

def record_zillowProperty(zillowProperty):
    try:
        host = 'localhost'
        database = 'zillow_objects'
        user = 'postgres'
        password = 'icuv371fhakletme'
        
        
        connection = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
        )
        
        cursor = connection.cursor()
        
        
        #testing area
        zillowProperty = {'amount': '49060', 
                      'valueChange': '-114', 
                      'low': '18152', 
                      'high': '81440', 
                      'percentile': '98', 
                      'zindexValue': 0, 
                      'last-updated': '01/07/2020', 
                      'street': '1027 Summit Dr', 
                      'zipcode': '90210', 
                      'city': 'Beverly Hills', 
                      'state': 'CA', 
                      'latitude': '34.086603', 
                      'longitude': '-118.420344', 
                      'FIPScounty': '6037', 
                      'useCode': 'SingleFamily', 
                      'taxAssessmentYear': '2019', 
                      'taxAssessment': '1.27449E7',
                      'yearBuilt': '1946', 
                      'lotSizeSqFt': '29938', 
                      'finishedSqFt': '6456', 
                      'bathrooms': '8.0', 
                      'bedrooms': '5', 
                      'lastSoldDate': '01/01/2018', 
                      'lastSoldPrice': '26500', 
                      'zpid': '20522179'
        }
        
        #converting scientific notation
        float(zillowProperty['taxAssessment'])
            
        
        
        #combining long/lat into Point datatype
        #SRID = 4326 (for geography datatype)
        zillowProperty['longitude_latitude'] = 'SRID=4326;POINT(%s %s)' % (zillowProperty['longitude'], zillowProperty['latitude'])
        del zillowProperty['longitude']
        del zillowProperty['latitude']
        
        #funneling dict into sql statement
        q2 = sql.SQL("INSERT INTO zillow_property ({}) values ({})").format(
                  sql.SQL(', ').join(map(sql.Identifier, zillowProperty)),
                  sql.SQL(', ').join(map(sql.Placeholder, zillowProperty)
        ))
        
        
        print('Inserting zillow Property...')
        cursor.execute(q2, zillowProperty)
        
        
        
        connection.commit()
        print('Property committed')
        
    
    
    
    
    
    
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
    finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
