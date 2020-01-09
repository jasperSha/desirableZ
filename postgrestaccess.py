import psycopg2
from psycopg2 import sql
#pipeline set up for funneling addresses into postgres database


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
                      'taxAssessment': '1.27449E7', #also need to format into int
                      'yearBuilt': '1946', 
                      'lotSizeSqFt': '29938', 
                      'finishedSqFt': '6456', 
                      'bathrooms': '8', #need to format into int, originally float
                      'bedrooms': '5', 
                      'lastSoldDate': '01/01/2018', 
                      'lastSoldPrice': '26500', 
                      'zpid': '20522179'
    }
    
    #combining long/lat into Point datatype
    zillowProperty['longitude_latitude'] = 'SRID=4326;POINT(%s %s)' % (zillowProperty['longitude'], zillowProperty['latitude'])
    del zillowProperty['longitude']
    del zillowProperty['latitude']
    
    #funneling dict into sql statement
    q2 = sql.SQL("INSERT INTO zillow_property ({}) values ({})").format(
                  sql.SQL(', ').join(map(sql.Identifier, zillowProperty)),
                  sql.SQL(', ').join(map(sql.Placeholder, zillowProperty)
        ))
    # print(q2.as_string(connection))
    cursor.execute(q2, zillowProperty)
    
    
    
    connection.commit()
    #SRID = 4326 (for geography datatype)

#    update_sql = ("""
#                  
#                  UPDATE zillow_property
#                  SET longitude_latitude = 'SRID=4326;POINT(-118.420344 34.086603)'
#                  WHERE zpid = 20522179;
#              
#                  """)
#    cursor.execute(update_sql)
#    connection.commit()
    
    
    
    
    

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
