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
        type(private/public),
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

def find_nearby_schools(key, state, city, school, limit=1):
    try:
        url = 'https://api.greatschools.org/schools/nearby'
        params = {
            'key': key,
            'state': state,
            'city':city,
            'limit':limit
            }
        attributes = vars(school)
        list_of_schools = []
        
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('finding schools near..', city)



        for attr in attributes:
            for child in root.iter('%s'%attr):
                attributes['%s'%attr] = child.text
                school.update(**attributes)
                list_of_schools.append(school)

        return list_of_schools
            
    except(Exception, requests.ConnectionError):
        print('Connection error..')
        
        

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


def school_profile(gsID, key, state):
    try:
        url = 'https://api.greatschools.org/schools/%s/%s'%(state, gsID)
        params = {
            'key': key
            }
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking up school associated with', gsID)
        
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')
        
        
def school_search(key, state, query, levelCode='', limit=1):
    try:
        url = 'https://api.greatschools.org/search/schools/'
        params = {
            'key':key,
            'state':state,
            'q':query,
            'levelCode':levelCode, #e.g. 'elementary-schools' or 'middle-schools'
            'limit':limit
            }
        
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        print('looking for:', query)
        
        
        for child in root.iter('name'):
            print(child.text)
    except (Exception, requests.ConnectionError):
        print('Connection error...')
        
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


school_attributes = {
        'gsId':'',
        'name':'',
        'public_private':'',
        'gradeRange':'',
        'enrollment':'',
        'gsRating':0, #(1-10 scale; 1-4: below average, 7-10: above average),
        'city':'',
        'state':'',
        # 'districtId':0,
        # 'district':'',
        # 'districtNCESId':0,
        # 'address':'',
        # 'ncesID':0,
        # 'lat':0,
        # 'lon':0,
        
        # 'overviewLink':'',
        # 'ratingsLink':'',
        # 'reviewsLink':'',
        # 'distance':0, #(from center?),
        # 'schoolStatsLink':''
    }
school = zillowObject.School(school_attributes)


print(find_nearby_schools(key, state, city, school, limit=10))






        
        
        
        
        
        
        
        
        
        
        


    
