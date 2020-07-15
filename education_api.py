import os
from dotenv import load_dotenv
load_dotenv()

""" 
KEYS:
    ZILLOW_API_KEY
    GREATSCHOOLS_API_KEY
    SOCRATA_CRIME_DATA_KEY


"""
key = os.getenv('GREATSCHOOLS_API_KEY')


import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import io
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from geoalchemy2.shape import to_shape



from zillowObject import zillowObject    

""" 
For browsing schools to find their gsID,
use find_nearby_schools() or browse_schools()

school census data provides data on ethnic distribution



For call:
    tags:
        gsId,
        name,
        type(private/public/charter),
        gradeRange,
        enrollment,
        gsRating (1-10 scale; 1-4: below average, 7-10: above average),
        city,
        state,
        districtId,
        district,
        districtNCESId,
        address,
        ncesID,
        lat,
        lon,
        
        overviewLink,
        ratingsLink,
        reviewsLink,
        distance (from center?),
        schoolStatsLink

"""
#list of schools closest to center of city, or lat/lon selected
def find_nearby_schools(key, state, city, school_attributes, limit=1):
    try:
        """ 
        only this one returns the gsRating
        """
        
        url = 'https://api.greatschools.org/schools/nearby'
        params = {
            'key': key,
            'state': state,
            'city':city,
            'limit':limit,
            'radius':10000
            }
        school_profiles = []
        
     
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('finding schools by gsID near..', city)
        
            
        for child in root.findall('school'):
            school = zillowObject.School(school_attributes)
            
            # for attribute in school:
            #     if child.find('%s'%attribute).text is not None:
            #         school['%s'%attribute] = child.find('%s'%attribute).text
            
            if child.find('gsRating') is None:
                #missing data schools need manual entry due to null fields
                school['type'] = child.find('type').text
                school['gsId'] = child.find('gsId').text
                # school['district'] = child.find('district').text
                school['gsRating'] = ''
                school['lat'] = child.find('lat').text
                school['lon'] = child.find('lon').text
                school['name'] = child.find('name').text
                
            else:  
                for attribute in school:
                    school['%s'%attribute] = child.find('%s'%attribute).text
            
            school['longitude_latitude'] = 'SRID=4326;POINT(%s %s)' % (school['lon'], school['lat'])
            del school['lat']
            del school['lon']
            
            school_profiles.append(school)
    
        return school_profiles
            
    except requests.exceptions.ConnectionError as ece:
        print("Connection Error:", ece)
    except requests.exceptions.Timeout as et:
        print("Timeout Error:", et)
    except requests.exceptions.RequestException as e:
        print("Some Ambiguous Exception:", e)
        
        
#list of all schools in a city
def browse_schools(key, state, city):
    try:
        url = 'https://api.greatschools.org/schools/%s/%s'%(state,city)
        params = {
            'key': key
            }
            
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('searching for schools in area..')
        
        
        for child in root.iter('name'):
            print(child.text)
    except requests.exceptions.ConnectionError as ece:
        print("Connection Error:", ece)
    except requests.exceptions.Timeout as et:
        print("Timeout Error:", et)
    except requests.exceptions.RequestException as e:
        print("Some Ambiguous Exception:", e)

#profile data for a specific school
def school_profile(key, state, gsID):
    try:
        url = 'https://api.greatschools.org/schools/%s/%s'%(state, gsID)
        params = {
            'key': key
            }
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking up school associated with', gsID)
 
        for child in root.iter():
            print(child.tag, child.text)
        
        # for location in root.iter('lat'):
        #     school['lat'] = location.text
        # for location in root.iter('lon'):
        #     school['lon'] = location.text
    except requests.exceptions.ConnectionError as ece:
        print("Connection Error:", ece)
    except requests.exceptions.Timeout as et:
        print("Timeout Error:", et)
    except requests.exceptions.RequestException as e:
        print("Some Ambiguous Exception:", e)
        
        


        
#returns list of schools based on query    
def school_search(key, state, query, levelCode='', limit=2):
    try:
        url = 'https://api.greatschools.org/search/schools/'
        params = {
            'key':key,
            'state':state,
            'q':query, #as in the name of the school, e.g. Alameda High School
            # 'levelCode':'', #e.g. 'elementary-schools' or 'middle-schools'
            'limit':limit
            }
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking for:', query)
        
        
        for child in root.iter('gsRating'):
            print(child.text)
        for child in root.iter('name'):
            print(child.text)
    except requests.exceptions.ConnectionError as ece:
        print("Connection Error:", ece)
    except requests.exceptions.Timeout as et:
        print("Timeout Error:", et)
    except requests.exceptions.RequestException as e:
        print("Some Ambiguous Exception:", e)
  
#list of most recent reviews for a school      
def school_reviews(key, state, city, gsID, topicID=None, school='school',limit=100):   
    try:
        
        if(school=='school'):
            #review for a specific school by gsID
            url = 'https://api.greatschools.org/reviews/school/%s/%s'%(state, gsID)
        else:
            #reviews for all schools in particular city
            url = 'https://api.greatschools.org/reviews/city/%s/%s'%(state, city)
        
        params = {
            'key':key,
            # 'topicID':topicID,
            'limit': limit
            }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking up reviews')
        
        for child in root.iter():
            print(child.text)
    except requests.exceptions.ConnectionError as ece:
        print("Connection Error:", ece)
    except requests.exceptions.Timeout as et:
        print("Timeout Error:", et)
    except requests.exceptions.RequestException as e:
        print("Some Ambiguous Exception:", e)




#list of school districts        
def browse_districts(key, state, city):
    try:
        url = 'https://api.greatschools.org/districts/%s/%s'%(state, city)
        params = {
            'key':key
            }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('finding list of school districts in', city)
        
        for child in root.iter('name'):
            print(child.text)
    except requests.exceptions.ConnectionError as ece:
        print("Connection Error:", ece)
    except requests.exceptions.Timeout as et:
        print("Timeout Error:", et)
    except requests.exceptions.RequestException as e:
        print("Some Ambiguous Exception:", e)


city = 'Los Angeles'
state = 'CA'
query = '90063'

school_attributes = {
        'gsId':'',
        'type':'',
        'name':'',
        # 'public_private':'',
        # 'gradeRange':'',
        # 'enrollment':'',
        'gsRating':'', #(1-10 scale; 1-4: below average, 7-10: above average),
        # 'city':'',
        # 'state':'',
        # 'districtId':0,
        # 'district':'',
        # 'districtNCESId':0,
        # 'address':'',
        # 'ncesID':0,
        'lat':'',
        'lon':''        
        # 'overviewLink':'',
        # 'ratingsLink':'',
        # 'reviewsLink':'',
        # 'distance':0, #(from center?),
        # 'schoolStatsLink':''
    }



os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
schools_df = gpd.read_file('greatschools/joined.shp')

schools = schools_df['name'].values.tolist()
gsIDs = schools_df['gsId'].values.tolist()
school_type = schools_df['type'].values.tolist()

# for i, j, k in list(zip(schools, gsIDs, school_type)):
#     print(j, ': ', i, ' type: ', k)


# census = []
# for school in schools:
#     traits = []
#     traits.append(school['gsId'])
#     traits.append(school['gsRating'])
#     traits.append(school['type'])
#     # traits.append(school['district'])
#     traits.append(school['name'])
#     # school['longitude_latitude'] = 'SRID=4326;POINT(%s %s)'%(school['lon'], school['lat'])
#     traits.append(school['longitude_latitude'])
#     # traits.append(school['lon'])
#     # traits.append(school['lat'])
#     census.append(traits)

# #see all panda columns
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

# schools_df = pd.DataFrame.from_records(census, columns=['gsId','gsRating','type','name','longitude_latitude'])
# schools_df.drop_duplicates
# schools_df.sort_values(by='name')
# print(schools_df.head(), schools_df.shape)

# # # print(schools_df.loc[schools_df['gsId']=='10946'])



# schools_df['longitude_latitude'] = schools_df['longitude_latitude'].apply(lambda x: x[10:].strip())
# schools_df['longitude_latitude'] = schools_df['longitude_latitude'].apply(wkt.loads)
# schools_gdf = gpd.GeoDataFrame(schools_df, geometry='longitude_latitude')

# print(schools_gdf.head(), schools_gdf.shape)


# schools_gdf.to_file('schools.gpkg', driver='GPKG')




# engine = create_engine('postgresql+psycopg2://postgres:icuv371fhakletme@localhost:5432/zillow_objects')

# schools_df.head(0).to_sql('la_county_education', engine, if_exists='append', index=False)

# conn = engine.raw_connection()
# cur = conn.cursor()
# output = io.StringIO()
# schools_df.to_csv(output,sep='\t', header=False, index=False)
# output.seek(0)
# cur.copy_from(output, 'la_county_education', null="")
# conn.commit()
    

# geo = []
# for school in schools:
#     point = []
#     point.append(school['lon'])
#     point.append(school['lat'])
#     geo.append(point)

# df = pd.DataFrame.from_records(geo, columns=['longitude', 'latitude'])
# print(df.head())

# #defining limits of mapping area

        
        
        
        
        
        


    
