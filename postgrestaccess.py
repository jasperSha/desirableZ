import psycopg2
from psycopg2 import sql
from config import config
#pipeline set up for funneling addresses into postgres database

def record_zillowProperty(zillowProperty):
    try:
        
        params = config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
       
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
        
        conn.commit()
        print('Property committed')
        
        cursor.close()
        conn.close()
    
    
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
    finally:
    #closing database connection.
        if(conn):
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")
            
            
def record_LA_addresses(parsed_address_list):
    try:
       
        params = config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
        
       
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
                      sql.SQL(', ').join(map(sql.Placeholder, parsed_address))
                      
                      )
        
            cursor.execute(q2, parsed_address)
            conn.commit()  
            
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
    finally:
    #closing database connection.
        if(conn):
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")
            
            

def pull_address_data():
    try:
       
        params = config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
    
        cursor.execute("""SELECT * FROM raw_address
                          OFFSET 254309;
        
                       """)
        addresses = cursor.fetchall()
        
        return addresses
    
    
    
    except(Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
    finally:
    #closing database connection.
        if(conn):
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")
    
    

    
    
            

