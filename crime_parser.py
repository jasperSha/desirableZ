import requests
import json
from sodapy import Socrata #module specific to socrata data site


#only appToken needed to prevent throttle limits
appToken = 'WutGZxIN3jEBDOZEqrzL5v82Q'


#currently heavily restricted on these datasets



crime_types = [
    'Assault',
    'Theft',
    'Assault with Deadly Weapon'
    ]

client = Socrata('data.lacity.org',
                  appToken
                  )
endpoints = [
    'tt5s-y5fc',#national police department crime rates
    'yi8n-dgju' #LA crime 2010 to present
    
    
    ]

results = client.get('63jg-8b9z', limit=2000)


for dictionary in results:
    print(dictionary['location'])