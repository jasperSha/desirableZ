from sodapy import Socrata #module specific to socrata data site
import pandas as pd



#only appToken needed to prevent throttle limits
appToken = 'WutGZxIN3jEBDOZEqrzL5v82Q'
lacityData = '63jg-8b9z'

#currently heavily restricted on these datasets
client = Socrata('data.lacity.org',
                  appToken
                  )
lacityData_categories = [
    'dr_no', #division record
    'date_occ',
    'date_rptd',
    
    'area',
    'area_name',
    'location',
    'cross_street',
    
    'lat',
    'lon',
    
    'crm_cd', #crime code
    'crm_cd_1', #crime code for severity: 1 most serious, 4 least serious
    'crm_cd_2',
    'crm_cd_3',
    'crm_cd_4',
    
    'crm_cd_desc',
    'mocodes',
    
    'vict_descent',
    'vict_age',
    'vict_sex',
    
    'premis_cd', #location or structure
    'premis_desc',
    
    'weapon_desc', #weapon description
    'weapon_used_cd',
    
    'status',
    'status_desc',
     
    ]

# endpoints = [
#     'tt5s-y5fc',#national police department crime rates
#     'yi8n-dgju' #LA crime 2010 to present
    
    
#     ]

results = client.get(lacityData, limit=2000)
results_df = pd.DataFrame.from_records(results)


print(results_df.columns)
    
# print(results_df['crm_cd_1'].value_counts())