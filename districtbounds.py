import os
from shapely.geometry import Polygon, Point, MultiPolygon
import geopandas as gpd

#read zillow property
os.chdir('/Users/Jasper/Documents/HousingMap/R_data/rental/')
properties = gpd.read_file('property.gpkg')

#read school district bounds
os.chdir('/Users/Jasper/Documents/HousingMap/R_data/schools/School_District_Boundaries')
lausd = gpd.read_file('School_District_Boundaries.shp')


#assing districts to schools
os.chdir('/Users/Jasper/Documents/HousingMap/R_data/schools/')
schools = gpd.read_file('schools.gpkg')


districts = list(lausd['DISTRICT'])
# print(districts)

polygons = list(lausd.geometry)

locales = dict(zip(districts, polygons))

def check_poly(point):
    for key, values in locales.items():
        if values.contains(point):
            return key
        
        
schools['DISTRICT'] = schools['geometry'].apply(check_poly)
schools.to_file('schools_distr.gpkg', layer='school', driver='GPKG')

# properties['DISTRICT'] = properties['geometry'].apply(check_poly)


# properties.to_file("property_distr.gpkg", layer='property', driver="GPKG")
