import csv
from postgrestaccess import record_LA_addresses as rla

#header of csv is
# LON, LAT, NUMBER, STREET, UNIT, CITY, DISTRICT, REGION, POSTCODE, ID, HASH
def address_parse(filename):
    
    with open(filename, 'rt', encoding='utf-8') as csv_file:
        print('opening file..')
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        
        
        #number of rows 2766584
        
        address_list = []
        repeats = []
        
        address_count = 0
        
        for skip in range(3): #skipping garbage data
            next(csv_reader)
        num =''
        
        for row in csv_reader:
            if line_count == 2766584:
                break
            
            address = {
            'number_street':'',
            'city_state':''
            }
            
            
            #have to refactor; set len(repeats) to at most 10 for checking
            #after 10 addresses, reset the repeat list
            
            if len(repeats) >=20:
                del repeats[:]
           
            city = row[5]            
            street = row[3]
            num = row[2] + ' ' + street
            
            if num not in repeats: #checking on repeats makes this N^2 i think
                address_count += 1
                address.update(number_street = num)
                address.update(city_state = ('%s CA'%city.strip()))
                
                address_list.append(address)
                repeats.append(num)
                print(address_count, ' addresses added')
            
            line_count += 1
        print('address list constructed, returning..')
        return address_list


pathname = '/Users/Jasper/Documents/Addresses/AddressHandling/AddressLookUp/AddressWest/us/ca/'
path = ('%slos_angeles.csv' % pathname)


x = address_parse(path)

rla(x)    
    
    
    
    