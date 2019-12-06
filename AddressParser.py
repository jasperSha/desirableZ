import csv
import pickle
import os
import glob

#header of csv is
# LON, LAT, NUMBER, STREET, UNIT, CITY, DISTRICT, REGION, POSTCODE, ID, HASH
def address_parse(filename):
    
    with open(filename, 'rt', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        address = {}
        s =''
        for row in csv_reader:
            if line_count == 0 or line_count == 1: #first two lines null
                line_count += 1
                continue
            if line_count == 2:
                line_count += 1
                continue
            if line_count == 500:
                break
            d = row[3].replace(' ','+')#number+street
            f = row[5].replace(' ','+')
            s = row[2] + '+' + d 
            if s not in address:
                address.update({('%s'%s):('%s+CA'%f.strip())})
                #row[5] is the name of city, stored as dict value, row 0, 1 = long/lat
                line_count += 1
            else:
                line_count += 1
                continue
        #address returned as ['number, street':'city, LONG/LAT']
        return address


city_addresses = []
pathname = '/Users/Jasper/Documents/Addresses/AddressHandling/AddressLookUp/AddressWest/us/ca/'
path = ('%slos_angeles.csv' % pathname)


x = address_parse(path)

for key, value in x.items():
    print('address:', key, 'city:', value)
##for file in glob.glob(path):
##    x = address_parse(file)
##    city_addresses.append(x)



##filename = ('%s'%address[0])
##outfile = open(filename, 'wb')
##pickle.dump(address, outfile)
##outfile.close()
##
##infile = open('ANAHEIM', 'rb')
##new_address = pickle.load(infile)
##infile.close()

