import requests
import xml.etree.ElementTree as ET

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
        lon
        
        overviewLink,
        ratingsLink,
        reviewsLink,
        distance (from center?),
        schoolStatsLink

"""
#list of schools closest to center of city, or lat/lon selected
def find_nearby_schools(key, state, city, limit=1):
    try:
        url = 'https://api.greatschools.org/schools/nearby'
        params = {
            'key': key,
            'state': state,
            'city':city,
            'limit':limit,
            'radius':50
            }
        schools_by_gsID = []
        
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('finding schools by gsID near..', city)

        
        for child in root.iter('gsId'):
            schools_by_gsID.append(child.text)

        

        return schools_by_gsID
            
    except(Exception, requests.ConnectionError):
        print('Connection error..')
        
        
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
    except (Exception, requests.ConnectionError):
        print('Connection error...')

#profile data for a specific school
def school_profile(gsID, key, state):
    try:
        url = 'https://api.greatschools.org/schools/%s/%s'%(state, gsID)
        params = {
            'key': key
            }
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking up school associated with', gsID)
        
        
        for child in root.iter('gsRating'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')
        
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
    except (Exception, requests.ConnectionError):
        print('Connection error...')
  
#list of most recent reviews fora school      
def school_reviews(key, state, city, topicID, gsID, school='school',limit=1):   
    try:
        #school selects specific school(by gsID) or all schools in city
        #alter url based on components
        if(school=='school'):
            url = 'https://api.greatschools.org/reviews/school/%s/%s'%(state, gsID)
        else:
            url = 'https://api.greatschools.org/reviews/city/%s/%s'%(state, city)
        
        params = {
            'key':key,
            'topicID':topicID,
            'limit': limit
            }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking up reviews')
        
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')


#list of topics available for topical reviews        
def review_topics(key):
    try:
        url = 'https://api.greatschools.org/reviews/reviewTopics'
        
        params = {
            'key': key
            }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('listing all portential review topics...')
        
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')
        


#returns census data for a specific school        
def school_census_data(key, state, gsID):
    try:
        url = 'https://api.greatschools.org/school/census/%s/%s'%(state, gsID)
        
        params = {
            'key': key
            }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('finding census data on school with gsID:', gsID)
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')


#returns info in a city        
def city_overview(key, state, city):
    try:
        url = 'https://api.greatschools.org/cities/%s/%s'%(state, city)
        params = {
            'key': key
            }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('finding overview of', city)
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')


#returns list of cities near specified city
def nearby_cities(key, state, city, radius=15):
    try:
        url = 'https://api.greatschools.org/cities/nearby/%s/%s'%(state, city) 
        params = {
            'key':key,
            'radius':radius #default 15, range 1-100
            }
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('searching for cities within', radius, 'miles of', city)
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')


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
    except (Exception, requests.ConnectionError):
        print('Connection error...')


key = 'e5fef655bc40261cbfddccbf0888b926'
city = 'Los-Angeles'
state = 'CA'
query = '90063'

school_attributes = {
        'gsId':'',
        # 'name':'',
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
        'lat':0,
        'lon':0,
        
        # 'overviewLink':'',
        # 'ratingsLink':'',
        # 'reviewsLink':'',
        # 'distance':0, #(from center?),
        # 'schoolStatsLink':''
    }


# schools =[]
#LA county is about 5500 schools
list_of_schools = find_nearby_schools(key, state, city, 10)


school_objects = []
    
for gsId in list_of_schools:
    school = zillowObject.School(school_attributes)
    school['gsId'].update(gsId)
    school_objects.append(school)

print(len(school_objects))




    

# print(len(schools))
#total schools found in 50 mile radius of LA county center is ~5508
# school_search(key, state, query)
        
        
        
        
        
        
        
        
        


    
