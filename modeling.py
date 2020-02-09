from config import config
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

params = config()
conn = psycopg2.connect(**params)

#creating ORM engine
engine = create_engine('postgresql://',creator=conn, echo=True)
Session = sessionmaker(bind=engine)

#session Object created
session = Session()

