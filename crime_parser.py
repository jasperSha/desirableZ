import requests
import json
import fnmatch


keyID = '52xjxy3zffcgnbf7mcatgduc0'
keySecret = '5w6zx1pg22lf1c8lgki6jepgl5amkkofrtzabhrnbnnmlio6v2'

headers = {
    'keyID': keyID,
    'keySecret': keySecret
    }

crime_types = [
    'Assault',
    'Theft',
    'Assault with Deadly Weapon'
    ]


kingCountySheriffs = requests.get('https://moto.data.socrata.com/resource/p6kq-vsa3.json', headers=headers)

##data = json.loads(kingCountySheriffs.text)
##output = ''
##for i in range(0, 100):
##    points = ''
##    points = data[i]['latitude']
##    points += data[i]['longitude']
##    
##    output = (data[i]['incident_datetime'][0:7] + ' '  + points  + '  '  + data[i]['incident_type_primary'][4:])
##    print(output)

morganHill = requests.get('https://moto.data.socrata.com/resource/s9ji-4jh6.json', headers=headers)

data = json.loads(morganHill.text)
output=''
for i in range(0, 10):
    points = ''
    points = data[i]['latitude'] + data[i]['longitude']
    
    print(data[i]['incident_datetime'][0:7] + ' ' + points + data[i]['incident_description'])

