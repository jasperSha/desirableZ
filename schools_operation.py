import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from cloud_geo import geocode_crime
import numpy as np


def read_shapefile(shp_path):
	"""
	Read a shapefile into a Pandas dataframe with a 'coords' column holding
	the geometry information. This uses the pyshp package
	"""
	import shapefile

	#read file, parse out the records and shapes
	sf = shapefile.Reader(shp_path)
	fields = [x[0] for x in sf.fields][1:]
	records = sf.records()
	shps = [s.points for s in sf.shapes()]

	#write into a dataframe
	df = pd.DataFrame(columns=fields, data=records)
	df = df.assign(coords=shps)

	return df



#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None


schools = gpd.read_file('/home/jaspersha/Projects/HeatMap/desirableZ/schools.gpkg')

# print(schools.head())
# print(schools.shape)
LAUSD_df = gpd.read_file('/home/jaspersha/VirtualSharedFolder/LAUSD_BOUNDS/School_District_Boundaries.shp')

district_cols = [col for col in districts_df.columns if col in ['DISTRICT', 'geometry']]
districts_df = districts_df[district_cols]

# print(districts_df.head())

schools.crs = 'EPSG:4326'
districts_df.crs = 'EPSG:4326'

joined = gpd.sjoin(schools, districts_df, op='within')

joined = joined.drop(['index_right'], axis=1)


print(joined.head(), joined.shape)


joined.to_file('schools_in_LAUSD_only.shp', driver='ESRI Shapefile')


# # schools_gdf.to_file('schools.gpkg', driver='GPKG')



start_date = '2015-01-01'
end_date = '2020-01-01'


# beverly = pd.read_csv('/home/jaspersha/Projects/HeatMap/GeospatialData/beverly_crime.csv')

# beverly_cols = [col for col in beverly.columns if col in ['Occurred From Date', 'Crime Type', 'Block Address']]
# beverly = beverly[beverly_cols]



# beverly = beverly.rename(columns={'Occurred From Date':'date_occ', 'Crime Type':'crm_cd_desc'})

# beverly['date_occ'] = beverly['date_occ'].apply(lambda x: x[:10].strip())
# beverly['date_occ'] = pd.to_datetime(beverly['date_occ'])
# mask = (beverly['date_occ'] > start_date) & (beverly['date_occ'] <= end_date)
# beverly = beverly.loc[mask]


# beverly = beverly.dropna()
# print(beverly.head())
# print(beverly.shape)
# print(beverly.dtypes)

# beverly['geometry'] = np.vectorize(geocode_crime)(beverly['Block Address'])
# beverly = beverly.drop(columns=['Block Address'])

# beverly.to_csv('beverly_wrangled.csv')

# print(beverly.head())



# beverly = pd.read_csv('/home/jaspersha/Projects/HeatMap/desirableZ/beverly_wrangled.csv')

# beverly = beverly.drop(columns={'Unnamed: 0'})
# beverly = beverly[['date_occ', 'crm_cd_desc', 'geometry']]
# beverly = beverly.dropna()

# beverly['geometry'] = beverly['geometry'].astype('string')
# beverly['date_occ'] = pd.to_datetime(beverly['date_occ'])
# print(beverly.isna().sum())
# beverly_df = pd.DataFrame(data=beverly)

# # beverly_df['geometry'] = beverly_df['geometry'].apply(wkt.loads)

# # print('beverly cols: ', beverly_df.columns)
# print('beverly head: ', beverly_df.head())
# # print('beverly dtypes: ', beverly_df.dtypes)




# crime = gpd.read_file('/home/jaspersha/Projects/HeatMap/GeospatialData/crime_sf.gpkg')

# crime['date_occ'] = pd.to_datetime(crime['date_occ'])

# start_date = '2015-01-01'
# end_date = '2020-01-01'

# mask = (crime['date_occ'] > start_date) & (crime['date_occ'] <= end_date)
# crime = crime.loc[mask]
# crime = crime[['date_occ', 'crm_cd_desc', 'geometry']]

# # print('crime dtypes: ', crime.dtypes)
# # print('crime cols: ', crime.columns)
# # print('crime head: ', crime.head())

# # # # crime.to_file('crime.shp', driver= 'ESRI Shapefile')




# santa_monica = pd.read_csv('/home/jaspersha/Projects/HeatMap/GeospatialData/santa_monica_crime.csv')
# santa_monica_cols = [col for col in santa_monica.columns if col in ['Incident Date', 'Call Type', 'Map Point']]
# santa_monica = santa_monica[santa_monica_cols]
# santa_monica = santa_monica.rename(columns={'Call Type': 'crm_cd_desc', 'Incident Date':'date_occ', 'Map Point':'geometry'})
# santa_monica['date_occ'] = santa_monica['date_occ'].apply(lambda x: x.replace('/', '-'))

# santa_monica['date_occ'] = pd.to_datetime(santa_monica['date_occ'])

# start_date = '2015-01-01'
# end_date = '2020-01-01'

# mask = (santa_monica['date_occ'] > start_date) & (santa_monica['date_occ'] <= end_date)
# santa_monica = santa_monica.loc[mask]


# santa_monica = santa_monica[['date_occ', 'crm_cd_desc', 'geometry']]
# santa_monica = santa_monica.dropna()

# # print(santa_monica.head(), santa_monica.shape)

# def fix(y):
#     x = y.split()
#     x[0], x[1] = x[1], x[0]
#     x[0] = x[0][:-1]
#     x[1] = x[1][1:-1]
#     return 'POINT (' + str(x[0]) + ' ' + str(x[1]) + ')'


# santa_monica['geometry'] = santa_monica['geometry'].astype('string')
# # beverly['geometry'] = np.vectorize(geocode_crime)(beverly['Block Address'])

# santa_monica['geometry'] = np.vectorize(fix)(santa_monica['geometry'])


# santa_monica_df = pd.DataFrame(data=santa_monica)
# print(santa_monica_df.head())

# santa_monica_df['geometry'] = santa_monica_df['geometry'].apply(wkt.loads)
# sm_gdf = gpd.GeoDataFrame(santa_monica_df, geometry='geometry')
# print(sm_gdf.head())
# sm_gdf.to_csv('santamonica.csv')

# sm_df = pd.read_csv('santamonica.csv')
# sm_df['geometry'] = sm_df['geometry'].apply(wkt.loads)
# smgdf = gpd.GeoDataFrame(sm_df, geometry='geometry')
# smgdf.crs = 'EPSG:4326'
# print(smgdf.head(), smgdf.crs)
# smgdf.to_file('santamonica.shp', driver='ESRI Shapefile')


# print('santa monica cols: ', santa_monica_df.columns)
# print('santa monica head: ' , santa_monica_df.head())
# print(santa_monica.shape)



# frames = [beverly_df, crime, santa_monica_df]

# true_combined_crime_df = pd.concat(frames)

# print(true_combined_crime_df.head(), '\n', true_combined_crime_df.dtypes, '\n', true_combined_crime_df.shape)

# true_combined_crime_df['date_occ'] = true_combined_crime_df['date_occ'].astype('string')
# true_combined_crime_df['crm_cd_desc'] = true_combined_crime_df['crm_cd_desc'].astype('string')

# full_crime = gpd.GeoDataFrame(true_combined_crime_df, geometry='geometry')
# full_crime.crs = 'EPSG:4326'
# # full_crime['geometry'] = pd.to_numeric(full_crime['geometry'], errors='coerce')
# print(full_crime.head(), '\n', full_crime.dtypes, '\n', full_crime.shape)
# print('full crime cols: ', full_crime.columns)

# # print(gpd.GeoDataFrame(full_crime).T.dtypes)
# # gpd.GeoDataFrame(full_crime).T.apply(pd.to_numeric, errors='ignore').to_file('full_crimes.shp')
# true_combined_crime_df.to_csv('fullcrime/full_crimes.csv')
# true_combined_crime_df.to_file('fullcrime/full_crimes.gpkg', layer='crime', driver='GPKG')


# fdf = pd.read_csv('fullcrime/full_crimes.csv')
# print(fdf.head())
# fdf['geometry'] = fdf['geometry'].apply(wkt.loads)
# crime_gdf = gpd.GeoDataFrame(fdf, geometry='geometry')
# crime_gdf.crs = 'EPSG:4326'
# crime_gdf.to_file('fullcrime/full_crimes.shp', driver='ESRI Shapefile')


