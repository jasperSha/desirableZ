import csv
import pickle
import os
import glob

def address_parse(filename):
    
    with open(filename, 'rt', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        address = {}
        s =''
        for row in csv_reader:
            if line_count == 0 or line_count == 1:
                line_count += 1
                continue
            if line_count == 50:
                break
            s = row[2] + ' ' + row[3]
            if s not in address:
                address.update({('%s'%s):('%s'%row[5])})
                line_count += 1
            else:
                line_count += 1
                continue
            return address

pathname = '/Users/Jasper/Documents/Addresses/AddressHandling/AddressLookUp/AddressWest/us/ca/'
path = ('%s*.csv' % pathname)

for file in glob.glob(path):
    x = address_parse(file)
    city_addresses.append(x)


print(city_addresses[0])

    




filename = ('%s'%address[0])
outfile = open(filename, 'wb')
pickle.dump(address, outfile)
outfile.close()

infile = open('ANAHEIM', 'rb')
new_address = pickle.load(infile)
infile.close()

