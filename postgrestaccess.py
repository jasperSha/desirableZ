import psycopg2
import io
from psycopg2 import sql
from sqlalchemy import create_engine
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
       
        #converting scientific notation
        float(zillowProperty['taxAssessment'])
                
        #combining long/lat into Point datatype
        #SRID = 4326 (for geography datatype)
        zillowProperty['longitude_latitude'] = 'SRID=4326;POINT(%s %s)' % (zillowProperty['longitude'], zillowProperty['latitude'])
        del zillowProperty['longitude']
        del zillowProperty['latitude']
        
        #funneling dict into sql statement
        q2 = sql.SQL("INSERT INTO zillow_property ({}) values ({})").format(
                  #reading the column names and matches to dictionary key
                  sql.SQL(', ').join(map(sql.Identifier, zillowProperty)),
                  #reading the column values
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
            
            
def record_LA_addresses(parsed_address_list):
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
       
        address_count = 0
        #list of 
        #funneling dict into sql statement
        for parsed_address in parsed_address_list:
            address_count += 1
            print('inserting address...', address_count)
            q2 = sql.SQL("INSERT INTO raw_address ({}) values ({})").format(
                      #reading the column names
                      sql.SQL(', ').join(map(sql.Identifier, parsed_address)),
                      #reading the values for corresponding column names
                      sql.SQL(', ').join(map(sql.Placeholder, parsed_address)
            ))
        
            cursor.execute(q2, parsed_address)
            connection.commit()  
            
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
    finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
            
            

