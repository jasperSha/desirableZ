import os
import pandas as pd

os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial')

zipcodes = pd.read_csv('zipcodelocations.csv', sep=';')

cols = ['Zip', 'geopoint']


