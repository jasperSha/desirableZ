from shapely.geometry import Point
import geopandas as gpd

def to_wkt(df):
    '''
    converts dataframe with separate longitude latitude columns to WKT format
    and returns as GeoDataFrame
    '''
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    df = df.drop(['longitude', 'latitude'], axis=1)
    
    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=geometry)
    return gdf
