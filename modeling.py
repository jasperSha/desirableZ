from modsim import *
from sodapy import Socrata #module specific to socrata data site
import pandas as pd

appToken = 'WutGZxIN3jEBDOZEqrzL5v82Q'
lacityData = '63jg-8b9z'

client = Socrata('data.lacity.org',
                  appToken
                  )


results = client.get(lacityData, limit=2000)
results_df = pd.DataFrame.from_records(results)

#staging area for using modsimpy to process these results


