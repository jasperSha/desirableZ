import os
from dotenv import load_dotenv
load_dotenv()

""" 
KEYS:
    ZILLOW_API_KEY
    GREATSCHOOLS_API_KEY
    SOCRATA_CRIME_DATA_KEY


"""
key = os.getenv('ZILLOW_API_KEY')


from zestimate import get_zestimate
from deepsearchresults import deep_search
from zillowObject.zillowObject import House
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
import requests
import time, signal, sys
terminate = False


#default property values
propertyDefaults = {
            'zpid':'',
            'amount': 0, # property value (rent)
            'valueChange': 0,
                #30-day
            'low': 0,
            'high': 0,
                #valuation range(low to high)
            'percentile': 0,
            'zindexValue': 0,

            'last-updated': '',
            'street': '',
            'zipcode': '',
            'city': '',
            'state': '',
            'latitude': '',
            'longitude': '',

            'FIPScounty': '',
            'useCode': '', 
                 #specifies type of home:
                 #duplex, triplex, condo, mobile, timeshare, etc
            'taxAssessmentYear': '',
                #year of most recent tax assessment
            'taxAssessment': 0,

            'yearBuilt': '',
            'lotSizeSqFt': 0,
            'finishedSqFt': 0,
            'bathrooms': 0,
            'bedrooms': 0,
            'lastSoldDate': '',
            'lastSoldPrice': 0
        }




def run_raw_address(citystatezip, address): #wrap into a function elsewhere?
    try:
        # zillowProperty = zillowObject.PropertyZest(propertyDefaults) #init zillowProperty as a zillowObject(init with given dict)
        zillow = House(propertyDefaults)
        #deep search call
        location = {'street' : address,
                    'city' : citystatezip}
        
        zillow.update(location)
        
        zillow.deep_search(key)


        if (zillow.zpid != ''): #making sure property is listed/not null, else continue
            #zestimate call
            zillow.get_zestimate(key)

            return zillow._values

        else:
            print('this address has no zpid, continuing..')            
            
    except:
        print("address failure somewhere")
        


def retrieve_zpid(prev_index):
    os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
    awz = gpd.read_file('zillowdb/zillowaws.shp')
    awz.drop_duplicates(subset=['street', 'city'],inplace=True)
    
    state = 'CA'
    streets = awz['street'].values.tolist()
    city = awz['city'].values.tolist()
    citystate = ['%s %s'%(x, state) for x in city]
    
    addresses = list(zip(streets, citystate))
    awz['zpid'] = ''
    os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ')
    
    for index, house in enumerate(addresses, start=prev_index):
        
        if (index-prev_index==200):
            time.sleep(30)
        
        url = 'https://zillow.com/webservice/GetDeepSearchResults.htm'
        parameters = {
            "zws-id": key,
            'address': house[0], #format: number+street+dr dr or drive ok
            'citystatezip': house[1] #format: city+state_abbreviation
        }
        response = requests.get(url, params=parameters)
        
        root = ET.fromstring(response.content)
        zpid = ''
        for element in root.iter('zpid'):
            zpid = element.text
        if zpid in ['', None]:
            print('Reached API limit; new index at: ', index)
            break
        else:
            print(zpid)
            awz.at[index, 'zpid'] = zpid
            
    
    awz = awz.iloc[prev_index:index]
    awz.to_csv('awzpid%s_%s.csv'%(prev_index,index))
    return

def signal_handling(signum, frame):
    global terminate
    terminate = True

def fix_shp(prev_count, stop_limit):
    #set signal handler for stopping to write file
    signal.signal(signal.SIGINT, signal_handling)
    

    os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/')
    awz = gpd.read_file('zillowdb/zillowaws.shp')
    awz.drop_duplicates(subset=['street', 'city'],inplace=True)
    
    state = 'CA'
    streets = awz['street'].values.tolist()
    city = awz['city'].values.tolist()
    citystate = ['%s %s'%(x, state) for x in city]
    
    addresses = list(zip(streets, citystate))
    new_awz = pd.DataFrame()
    for count, house in enumerate(addresses[prev_count:], start=prev_count):
        if count - prev_count == 4500:
            break
        if count == stop_limit:
            break
        if terminate:
            break
        zill = run_raw_address(house[1], house[0])
        new_awz = new_awz.append(zill, ignore_index=True)
    print(new_awz.tail())
    os.chdir('/home/jaspersha/Projects/HeatMap/GeospatialData/compiled_heatmap_data/houses_compiled/')
    new_awz.to_csv('awz_%s_%s.csv'% (prev_count, count), index=False)
    print('new count: ', count)
    return
        

if __name__=='__main__': 
    
    
    #limit is 4500 (~13.5k calls)    

    
    
    fix_shp(289500, 4500)
    
    
    
    
    
    
    