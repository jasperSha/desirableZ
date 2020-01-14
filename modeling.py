import matplotlib.pyplot as plt
from sodapy import Socrata #module specific to socrata data site
import pandas as pd
import numpy as np

appToken = 'WutGZxIN3jEBDOZEqrzL5v82Q'
lacityData = '63jg-8b9z'

client = Socrata('data.lacity.org',
                  appToken
                  )


results = client.get(lacityData, limit=5000)
data = pd.DataFrame.from_records(results)


#only the necessary columns
df = data[['date_occ','crm_cd_desc','weapon_desc','lon','lat']]

#datetime conversion
date_time = data[['date_occ']]
date_convert = pd.to_datetime(date_time, yearfirst=True)

#mapping severity of crime to time committed
time_df = data[['crm_cd_desc','time_occ']]


# time_df.plot(kind='scatter',x='time_occ', y='crm_cd_desc',color='red')

# plt.show()