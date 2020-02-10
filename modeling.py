from config import config
import psycopg2
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base



def connect():
    params = config()
    conn = psycopg2.connect(**params)
    return conn

#instantiating base for table class mapping
Base = automap_base()

#starting up ORM engine
engine = create_engine('postgresql://',creator=connect)

#reflecting the table classes to map the Base
Base.prepare(engine, reflect=True)


session = Session(engine)

#session created for database interface

for mappedclass in Base.classes:
    print(mappedclass)



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




