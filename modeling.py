from config import config
import psycopg2
from sqlalchemy import Column, Integer, String, Float, DATE
from sqlalchemy import create_engine, MetaData, Table, Column
from geoalchemy2.shape import to_shape
from geoalchemy2 import Geography
from geoalchemy2 import functions
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import geopandas
from shapely import wkt



#see all panda columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#defining county_school object for ORM queries
#matches name to column but can skip table columns

def connect():
    params = config()
    conn = psycopg2.connect(**params)
    return conn

    



def school_query():
    # #starting up ORM engine   
    engine = create_engine('postgresql://',creator=connect)
    
    #binding Session class to engine
    Session = sessionmaker(bind=engine)
    
    #instantiating Session as object
    session = Session()
    Base = declarative_base()
    
    class County_School(Base):
        __tablename__ = 'la_county_education'
        
        gsId = Column(Integer, primary_key=True)
        gsRating = Column(Integer)
        type = Column(String)
        name = Column(String)
        longitude_latitude = Column(Geography(geometry_type='POINT', srid=4326))
        
        def __repr__(self):
            return "<County_School(gsId='%s', gsRating='%s')>"%(self.gsId, self.gsRating)
        
        
    
    fields = ['gsId', 'gsRating', 'name', 'longitude_latitude', 'type']
    
    records = session.query(County_School).all()
   
    #apply county_school columns to dataframe columns
    df = pd.DataFrame([{fn: getattr(f, fn) for fn in fields} for f in records])
    #lon/lat to POINT format
    df['longitude_latitude'] = df['longitude_latitude'].apply(lambda x: to_shape(x).to_wkt())
    df['longitude_latitude'] = df['longitude_latitude'].apply(wkt.loads)
    gdf_schools = geopandas.GeoDataFrame(df, geometry='longitude_latitude')
    
    gdf_schools.crs = 'EPSG:4326'
    return gdf_schools
    # gdf_schools.to_file("school_package.gpkg", layer='schools', driver="GPKG")
    

def crime_query():
    # #starting up ORM engine   
    engine = create_engine('postgresql://',creator=connect)
    
    #binding Session class to engine
    Session = sessionmaker(bind=engine)
    
    #instantiating Session as object
    session = Session()
    Base = declarative_base()
    
    class Crime_Spots(Base):
        __tablename__ = 'la_crime'
        
        dr_no = Column(Integer, primary_key=True)
        date_rptd = Column(DATE)
        date_occ = Column(DATE)
        time_occ = Column(String) #military
        area_name = Column(String)
        rpt_dist_no = Column(Integer)
        crm_cd_desc = Column(String) #for rating severity of the crime
        vict_descent = Column(String)
        vict_age = Column(Integer)
        vict_sex = Column(String)
        premis_cd = Column(Integer)
        weapon_desc = Column(String) #returns the status(abbreviated format)
        status = Column(String) #returns the status description
        status_desc = Column(String) #returns the weapon description
        longitude_latitude = Column(Geography(geometry_type='POINT', srid=4326))
    
    
    fields = ['date_occ', 'crm_cd_desc', 'longitude_latitude'] 
    for year in range(2019, 2020):
        records = session.query(Crime_Spots).filter(Crime_Spots.date_occ.between('%s-01-01'%year,'%s-12-31'%year)).all()
        df = pd.DataFrame([{fn: getattr(f, fn) for fn in fields} for f in records])
        #lon/lat to POINT format
        df['longitude_latitude'] = df['longitude_latitude'].apply(lambda x: to_shape(x).to_wkt())
        df['longitude_latitude'] = df['longitude_latitude'].apply(wkt.loads)
        df['date_occ'] = df['date_occ'].apply(lambda x: x.strftime('%Y-%m-%d'))
        gdf_crime = geopandas.GeoDataFrame(df, geometry='longitude_latitude') 
        gdf_crime.crs = 'EPSG:4326'
        return gdf_crime
        # gdf_schools.to_file("%s_crime.gpkg"%year, layer='crime', driver="GPKG")
    
    

def zillow_query():
     # #starting up ORM engine   
    engine = create_engine('postgresql://', creator=connect)
    
    #binding Session class to engine
    Session = sessionmaker(bind=engine)
    
    #instantiating Session as object
    session = Session()
    Base = declarative_base()
    
    class Zillow_Property(Base):
        __tablename__ = 'zillow_property'
        
        zpid = Column(Integer, primary_key=True)
        longitude_latitude = Column(Geography(geometry_type='POINT', srid=4326))
        zipcode = Column(Integer)
        city = Column(String)
        state = Column(String)
        valueChange = Column(Integer)
        yearBuilt = Column(Integer)
        lotSizeSqFt = Column(Integer)
        finishedSqFt = Column(Integer)
        lastSoldPrice = Column(Integer)
        amount = Column(Integer)
        taxAssessmentYear = Column(Integer)
        FIPScounty = Column(Integer)
        low = Column(Integer)
        high = Column(Integer)
        percentile = Column(Integer)
        zindexValue = Column(Integer)
        street = Column(String)
        lastSoldDate = Column(DATE)
        useCode = Column(String)
        bathrooms = Column(Float)
        bedrooms = Column(Integer)
        taxAssessment = Column(Float)
        def __repr__(self):
            return "<Zillow_Property(zpid='%s', Monthly Rental='%s')>"%(self.zpid, self.amount)
    
    fields = ['amount', 'longitude_latitude', 'zindexValue', 'useCode', 'finishedSqFt', 'lotSizeSqFt', 'low', 'high', 'percentile', 'bathrooms', 'bedrooms', 'taxAssessment']
    records = session.query(Zillow_Property).limit(1000).all()
    
    df = pd.DataFrame([{fn: getattr(f, fn) for fn in fields} for f in records])
    #lon/lat to POINT format
    df['longitude_latitude'] = df['longitude_latitude'].apply(lambda x: to_shape(x).to_wkt())
    df['longitude_latitude'] = df['longitude_latitude'].apply(wkt.loads)
    
    gdf_properties = geopandas.GeoDataFrame(df, geometry='longitude_latitude') 
    gdf_properties.crs = 'EPSG:4326'
    return gdf_properties
    # gdf_properties.to_file("property.gpkg", layer='property', driver="GPKG")


crime = crime_query()



"""
la crime_desc


CATEGORIES:
    FELONY:
        SUPPORTED BY INTENT TO KILL OR GRIEVOUSLY INJURE
        RAPE
        RATING: 5
        
    FELONY ASSAULT:
        AGGRAVATED
        DEADLY WEAPON
        
        RATING: 4
        
    FELONY:
        GRAND THEFT OR VANDALISM
        INTRUSION OF PERSONAL PROPERTY
        SIMPLE ASSAULT
        
        RATING: 3
    
    MISDEMEANOR:
        LESS SERIOUS, PUNISHABLE BY JAIL INSTEAD OF PRISON
        PETTY THEFT
        
        RATING: 2
        
        
    INFRACTIONS:
        AKA VIOLATIONS - TRAFFIC TICKETS, JAYWALKING, ETC
        RATING: 1


Violation of Court Order
RATING: 1

Criminal Homicide
RATING: 5

Vandalism - Felony ($400+)
RATING: 3

Miscellaneous
RATING: 1

Attempted Rape
RATING: 5

Shoplifting - Petty theft($950-)
RATING: 2

Burglary from vehicle
RATING: 3

Assault with Deadly Weapon, Aggravated Assault
RATING: 4

Theft-GRAND ($950+)
RATING: 3

Battery - Simple Assault
RATING: 3

Robbery
RATING: 3

Bomb Scare
RATING: 3

Theft from motor vehicle
RATING: 2

Child Neglect
RATING: 3

Intimate Partner - Aggravated Assault
RATING: 4

Intimate Partner - Simple Assault
RATING: 3

Theft Plain - Petty ($950-)
RATING: 2

Criminal Threats - No weapon displayed
RATING: 2

Vandalism- Misdemeanor ($399-)
RATING: 3

Arson
RATING: 4

Rape, Forcible
RATING: 5

Throwing Object at Moving Vehicle
RATING: 3

Child Abuse (physical) - Simple Assault
RATING: 3

Shots fired at Inhabited Dwelling
RATING: 4

Pickpocket
RATING: 1

Trespassing
RATING: 1

Vehicle - Stolen
RATING: 3

Document Forgery
RATING: 2

Battery with Sexual Contact
RATING: 4

Brandish Weapon
RATING: 3

Violation of Restraining Order
RATING: 2

Kidnapping
RATING: 4

Indecent Exposure
RATING: 3

Peeping Tom
RATING: 2

Theft of Identity
RATING: 2

Embezzlement
RATING: 2

Extortion
RATING: 2

Lewd Conduct
RATING: 2

Sexual Penetration w/foreign object
RATING: 4

Contempt of Court
RATING: 1

Letters, lewd
RATING: 2

SEX, UNLAWFUL
RATING: 3

BUNCO, GRAND THEFT
RATING: 3



date_occ

RATING AFFECTED BY DATE OCCURRED

FOR EACH YEAR, 


"""




