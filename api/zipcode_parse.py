import geopandas as gp
import pandas as pd
import os
import sys
import glob

#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None


def parse_gpkg():
    cwd = '/Users/Jasper/Documents/GIS_Database/addresses_by_zipcode/ZipCode_'
    
    zip_91354 = gp.read_file(cwd+'91772.gpkg')

    def parse_frame(row):
        try:
            addr_number = row['Number']
            addr_street = row['FullName']
            addr_city = row['LegalComm']
            
            address = str(addr_number)+ ' ' + addr_street, addr_city
            
            return address
        except Exception:
            pass
    
    zip_91354['new_address'] = zip_91354.apply(parse_frame, axis=1)
    addresses = []
    newdf = zip_91354['new_address'].iloc[::10]
    addresses = (list(newdf.values))
    
    new_addresses = []
    
    for i in addresses:
        number, city = i
        if city == 'Unincorporated':
            city = 'Los Angeles'
        new_addresses.append((number, city + ' CA'))
    return new_addresses
















