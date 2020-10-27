import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
register_matplotlib_converters()
from time import time

#reading the data and creating another dataframe for the stationary points
df_heater = pd.read_csv('heater.csv')
adj_df_heater = df_heater.iloc[0:109]

#showing the head of data
df_heater.head()
adj_df_heater.head()

#rename columns to something more understandable
df_heater.rename(columns={'Month':'month', 'heater: (United States)':'search_results'}, inplace=True)
adj_df_heater.rename(columns={'Month':'month', 'heater: (United States)':'search_results'}, inplace=True)

#convert date column to datetime type
df_heater['month'] = pd.to_datetime(df_heater.month)
adj_df_heater['month'] = pd.to_datetime(adj_df_heater.month)


#set date as index
df_heater.set_index('month', inplace=True)
adj_df_heater.set_index('month', inplace=True)


#show result
df_heater.head()
adj_df_heater.head()


#Plotting total
plt.figure(figsize=(10,4))
plt.plot(df_heater.search_results)
plt.title('Total "heater" Search Results Over Time', fontsize=22)
plt.ylabel('Search Results', fontsize=16)
plt.xlabel('Year', fontsize=16)
for year in range(2003,2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

#Plotting adjusted
plt.figure(figsize=(10,4))
plt.plot(adj_df_heater.search_results)
plt.title('Stationary "heater" Search Results Over Time', fontsize=22)
plt.ylabel('Search Results', fontsize=16)
plt.xlabel('Year', fontsize=16)
for year in range(2004,2014):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
    
    
#acf
acf_plot = plot_acf(df_heater.search_results, lags=100)
adj_acf_plot = plot_acf(adj_df_heater.search_results, lags=100)


#pacf
pacf_plot = plot_pacf(df_heater.search_results)
#we can see significant lag effects, 1,2,3,8,9,10,11,12,13,15,21
adj_pacf_plot = plot_pacf(adj_df_heater.search_results)
#we can see significant lag effects, 1,2,3,8,9,10,11,12,13,15,16,18,19,20

#infer frequency
#adj_df_heater = adj_df_heater.asfreq(pd.infer_freq(adj_df_heater.index))



#AR MODELS STARTING HERE
#get training and test set
train_end = datetime(2012,1,1)
test_end = datetime(2013,1,1)

train_data = adj_df_heater[:train_end]
test_data = adj_df_heater[train_end + timedelta(days=1):test_end]

#create the AR model use order (X,0) in ARMA, so only using the AR model
model = ARMA(train_data, order=(15,0))
start = time()
model_fit = model.fit()
end = time()
print('Model fitting time', end-start)
#Model fitting time AR(3) 0.154249906539917
#Model fitting time AR(7) 0.494765043258667
#Model fitting time AR(10) 2.5992798805236816
#Model fitting time AR(13) 5.2760021686553955

#summary of the model
print(model_fit.summary())


#get prediction start and end dates
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

#predictions and residuals
predictions = model_fit.predict(start = pred_start_date, end =pred_end_date)
residuals = test_data.search_results - predictions

#plotting predictions vs actual data
plt.figure(figsize=(10,4))
plt.plot(test_data)
plt.plot(predictions)
plt.legend(('Data', 'AR(15) Predictions'), fontsize=16)
plt.title('Heater search frequency', fontsize=20)
plt.ylabel('Search Frequency', fontsize=16)

#plotting residuals
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title('Residuals from AR(15) Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)
for year in range(2012,2013):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
    
    
#INDICATORS OF ACCURACY
#Calculating mean absolute percent error (on average, in terms of percentage how far is the predicted from the actual)
print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data.search_results)),5)*100, '%')
#AR(3) = 20.748 %
#AR(7) Mean Absolute Percent Error: 11.557 %
#AR(10) Mean Absolute Percent Error: 7.179 %
#AR(13) Mean Absolute Percent Error: 6.813 %
#AR(15) Mean Absolute Percent Error:7.338 %

#Calculating Root Mean Squared Error
print('Root Mean Squared Error:', round(np.sqrt(np.mean(residuals**2)),5))
#AR(3) = 9.11655
#AR(7) Root Mean Squared Error: 4.25513
#AR(10) Root Mean Squared Error: 3.24473
#AR(13) Root Mean Squared Error: 2.92978
#AR(15) Root Mean Squared Error: 3.0607

#predicting 2013 to 2014 
train_end_new = datetime(2013,1,1)
test_end_new = datetime(2015,1,1)

train_data_new = df_heater[:train_end_new]
test_data_new = df_heater[train_end_new + timedelta(days=1):test_end_new]

model_new = ARMA(train_data_new, order=(10,0))
start = time()
model_fit_new = model_new.fit()
end = time()
print('Model fitting time', end-start)

print(model_fit_new.summary())

pred_start_date_new = test_data_new.index[0]
pred_end_date_new = test_data_new.index[-1]

#predictions and residuals
predictions_new = model_fit_new.predict(start = pred_start_date_new, end =pred_end_date_new)
residuals_new = test_data_new.search_results - predictions_new

plt.figure(figsize=(10,4))
plt.plot(test_data_new)
plt.plot(predictions_new)
plt.legend(('Data', 'AR(10) Predictions'), fontsize=16)
plt.title('Heater search frequency 2013-2015', fontsize=20)
plt.ylabel('Search Frequency', fontsize=16)

plt.figure(figsize=(10,4))
plt.plot(residuals_new)
plt.title('Residuals from AR(10) Model for 2013-2015', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', linestyle='--', alpha=0.2)
for year in range(2013,2015):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
    
    

print('New Mean Absolute Percent Error:', round(np.mean(abs(residuals_new/test_data_new.search_results)),5)*100, '%')
#New Mean Absolute Percent Error: 16.993 %
print('New Root Mean Squared Error:', round(np.sqrt(np.mean(residuals_new**2)),5))
#New Root Mean Squared Error: 10.90079

