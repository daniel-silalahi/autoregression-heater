# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
register_matplotlib_converters()
from time import time

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

#change format from 2004-01 to 2004-01-01
df_heater = pd.read_csv('heater.csv')
df_heater.rename(columns={'Month':'month', 'heater: (United States)':'search_results'}, inplace=True)
df_heater['month'] = pd.to_datetime(df_heater.month)
df_heater
df_heater.set_index('month', inplace=True)

#read data
heater_search = pd.read_csv(df_heater, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)