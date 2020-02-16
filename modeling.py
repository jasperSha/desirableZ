from config import config
import psycopg2
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy import create_engine, MetaData, Table, Column
from geoalchemy2.shape import to_shape
from geoalchemy2 import Geography
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import feather

#defining county_school object for ORM queries
#matches name to column but can skip table columns

    
# class Zillow_Property(Base):
#     __tablename__ = 'zillow_property'
    
#     zpid = Column(Integer, primary_key=True)
#     longitude_latitude = Column(Geography(geometry_type='POINT', srid=4326))
#     zipcode = Column(Integer)
#     city = Column(String)
#     state = Column(String)
#     valueChange = Column(Integer)
#     yearBuilt = Column(Integer)
#     lotSizeSqFt = Column(Integer)
#     finishedSqFt = Column(Integer)
#     lastSoldPrice = Column(Integer)
#     amount = Column(Integer)
#     taxAssessmentYear = Column(Integer)
#     FIPScounty = Column(Integer)
#     low = Column(Integer)
#     high = Column(Integer)
#     percentile = Column(Integer)
#     zindexValue = Column(Integer)
#     street = Column(String)
#     lastSoldDate = Column(String)
#     useCode = Column(String)
#     bathrooms = Column(Float)
#     bedrooms = Column(Integer)
#     taxAssessment = Column(Float)
#     def __repr__(self):
#         return "<Zillow_Property(zpid='%s', Monthly Rental='%s')>"%(self.zpid, self.amount)
    
# class Crime_Spots(Base):
#     __tablename__ = 'la_crime'
    
#     dr_no = Column(Integer, primary_key=True)
#     date_rptd = Column(String)
#     date_occ = Column(String)
#     time_occ = Column(String)
#     area_name = Column(String)
#     rpt_dist_no = Column(Integer)
#     crm_cd_desc = Column(String) #for rating severity of the crime
#     vict_descent = Column(String)
#     vict_age = Column(Integer)
#     vict_sex = Column(String)
#     premis_cd = Column(Integer)
#     weapon_desc = Column(String) #returns the status(abbreviated format)
#     status = Column(String) #returns the status description
#     status_desc = Column(String) #returns the weapon description
#     longitude_latitude = Column(Geography(geometry_type='POINT', srid=4326))



def connect():
    params = config()
    conn = psycopg2.connect(**params)
    return conn

    



def crime_query():
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
        # type = Column(String)
        name = Column(String)
        longitude_latitude = Column(Geography(geometry_type='POINT', srid=4326))
        
        def __repr__(self):
            return "<County_School(gsId='%s', gsRating='%s')>"%(self.gsId, self.gsRating)
        
        
    count = 0
    geos = []
    
    df = pd.read_sql(session.query(County_School).order_by(County_School.gsId))
    print(df)
    for instance in session.query(County_School).order_by(County_School.gsId):
        if count == 10:
            break
        count += 1
        shply_geom = to_shape(instance.longitude_latitude)
        geos.append(shply_geom.to_wkt())
        print(instance)
    print(geos)
    
    
crime_query()
#radius 


""" 
zpid
AMOUNT (ZESTIMATE)
long/lat

valueChange (30 day zestimate change)
lotSizeSqFt
finishedSqFt
lastSoldPrice

low (valuation range)
high (valuation range)
percentile 
zindexValue

useCode
bathrooms
bedrooms


"""



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




