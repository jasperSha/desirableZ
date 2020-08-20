from sodapy import Socrata #module specific to socrata data site
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import io
import os
from dotenv import load_dotenv
load_dotenv()

""" 
KEYS:
    ZILLOW_API_KEY
    GREATSCHOOLS_API_KEY
    SOCRATA_CRIME_DATA_KEY


"""
appToken = os.getenv('SOCRATA_CRIME_DATA_KEY')

#la repo
lacityData = '63jg-8b9z'

#currently heavily restricted on these datasets
client = Socrata('data.lacity.org',
                  appToken
                  )

query = """ 
select
    dr_no,
    date_rptd,
    date_occ,
    time_occ,
    area_name,
    lat,
    lon,float
    rpt_dist_no,
    crm_cd_desc,
    vict_descent,
    vict_age,
    vict_sex,
    premis_cd,
    premis_desc,
    weapon_desc,
    status,
    status_desc
limit
    2000000

"""



""" 
integer
date
date
time without time zone
text
integer
text
text
integer
text
integer
text
text
text
text
geography

"""
results = client.get(lacityData, query=query)
results_df = pd.DataFrame.from_records(results)


# pd.set_option('max_colwidth', 100)
# pd.set_option('display.max_columns', 19)




for row in results_df.index:
    results_df.at[row, 'date_rptd'] = results_df.at[row,'date_rptd'][:10]
    results_df.at[row, 'date_occ'] = results_df.at[row, 'date_occ'][:10]
    results_df.at[row, 'time_occ'] = results_df.at[row, 'time_occ'][:2] + ':' + results_df.at[row, 'time_occ'][2:]
    
    if results_df.at[row, 'vict_age'] == '0':
        results_df.at[row,'vict_age'] = ""



#formatting long/lat to comply with srid
results_df['longitude_latitude'] = 'SRID=4326;POINT(' + results_df['lon'].map(str) + ' ' + results_df['lat'].map(str) + ')'
del results_df['lon']
del results_df['lat']

# print(results_df.head())

""" 

had to create table with column types already set and then set
if_exists='append' in order for the POSTGIS geography type to work.

"""

engine = create_engine('postgresql+psycopg2://postgres:icuv371fhakletme@localhost:5432/zillow_objects')



results_df.head(0).to_sql('la_crime', engine, if_exists='append', index=False)

conn = engine.raw_connection()
cur = conn.cursor()
output = io.StringIO()
results_df.to_csv(output,sep='\t', header=False, index=False)
output.seek(0)
cur.copy_from(output, 'la_crime', null="")
conn.commit()
    
    









