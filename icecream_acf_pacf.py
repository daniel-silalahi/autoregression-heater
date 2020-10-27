import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()

#reading the data
df_ice_cream = pd.read_csv('icecream.csv')
#df_ice_cream_2 = pd.read_csv('icecream.csv')

#showing the head of data
df_ice_cream.head()

#rename columns to something more understandable
df_ice_cream.rename(columns={'DATE':'date', 'IPN31152N':'index_production'}, inplace=True)
#df_ice_cream_2.rename(columns={'DATE':'date', 'IPN31152N':'index_production'}, inplace=True)


#convert date column to datetime type
df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
#df_ice_cream_2['date'] = pd.to_datetime(df_ice_cream_2.date)


#set date as index
df_ice_cream.set_index('date', inplace=True)
df_ice_cream_2.set_index('date', inplace=True)


#just get data from 2010 onwards
start_date = pd.to_datetime('2010-01-01')
#df = pd.DataFrame({'year': [2010, 2020],
#                    'month': [1, 1],
#                    'day': [1, 1]})
#ten_year_data = pd.to_datetime(df)
#ten_year_data_2 = pd.to_datetime('2010-01-01', '2020-01-01')
df_ice_cream = df_ice_cream[start_date:]
#df_ice_cream_2 = df_ice_cream_2[ten_year_data:]

#show result
df_ice_cream.head()

#Plotting
plt.figure(figsize=(10,4))
plt.plot(df_ice_cream.index_production)
plt.title('Ice Cream Production Index over Time', fontsize=22)
plt.ylabel('Production Index', fontsize=16)
plt.xlabel('Year', fontsize=16)
for year in range(2010,2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
    
#acf
acf_plot = plot_acf(df_ice_cream.index_production, lags=100)

#pacf
pacf_plot = plot_pacf(df_ice_cream.index_production)
#we can see significant lag effects, 1,2,3,7,10,11,13,16,19




