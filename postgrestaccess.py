import psycopg2
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
    password=password,
)

    cursor = connection.cursor()
    # Print PostgreSQL Connection properties
    print(connection.get_dsn_parameters(),"\n")

    # Print PostgreSQL version
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record,"\n")

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
