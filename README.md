# autoregression-heater
Using an AR model (a time series model) to predict the future frequency of search results for the word "heater" on google trends.

Time series is simply a series of data points ordered in time, such as the unemployment rate or the median price of university tuition over a fixed interval of time (eg monthly, yearly). Time series models such as AR, ARCH, GARCH, ARIMA are used on time series data, these data often arise when monitoring industrial processes or tracking corporate business metrics. Time series analysis accounts for data that may have an internal structure, such as seasonality, trend or autocorrelation that should be accounted for.

Time series models could be used to understand underlying structures in data and also to forecast or monitor future trends from previous data. More specifically, it can be used for Stock market analysis, Sales forecasting and budgetary analysis.

Time series data can be described as stationary or non-stationary.  A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. This allows models to be used outside of the range of data, allowing extrapolation. Non stationary data needs to be adjusted before prediction/analysis, a method of adjustment is by taking the “first difference”, which is the difference between a point and the point just before, this can tell us the volatility and we may be able to predict future volatility through analysis.

A dataset that we will predict the future results of will be introduced in the next section.

The dataset I will be using is the amount of search results for the word ‘heater’ over time in the United States with a time interval of 1 month, this data is available through google trends (https://trends.google.com/trends/explore?date=all&geo=US&q=heater)


