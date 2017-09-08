
# coding: utf-8

# In[2]:

import json
import requests
import quandl
import pandas as pd

from pandas.io.json import json_normalize



response = requests.get("https://min-api.cryptocompare.com/data/histoday?fsym=ETH&tsym=USD&allData=TRUE")


data = json.loads(response.text)
data2 = json_normalize(data)
json_data = data['Data']
df=pd.DataFrame(data['Data'])
df['Date'] = pd.to_datetime(df['time'],unit='s')
df.dtypes
df.astype(object)

# US Treasury Bond Futures
USTBonds = quandl.get('CHRIS/CME_US1')

# US Dollar Index Futures
USDIndexFutures = quandl.get('CHRIS/ICE_DX2')
    
# Inflation YOY - USA
USInflationRate = quandl.get('RATEINF/INFLATION_USA')

# Unemployment Level
USUnemp = quandl.get('FRED/LNU03000000')
    
# Effective Federal Funds Rate (Interest Rate)
USFedFundRate = quandl.get('FRED/DFF')

# United States GDP at Current Prices, LCU Billions
USGDP = quandl.get('ODA/USA_NGDP')


# In[41]:

from datetime import datetime


USTBonds = pd.DataFrame(USTBonds)
USTBonds['Date'] = USTBonds.index.values

USDIndexFutures = pd.DataFrame(USDIndexFutures)
USDIndexFutures['Date'] = USDIndexFutures.index.values

USInflationRate = pd.DataFrame(USInflationRate)
USInflationRate['Date'] = USInflationRate.index.values

USUnemp = pd.DataFrame(USUnemp)
USUnemp['Date'] = USUnemp.index.values

USFedFundRate = pd.DataFrame(USFedFundRate)
USFedFundRate['Date'] = USFedFundRate.index.values

USGDP = pd.DataFrame(USGDP)
USGDP['Date'] = USGDP.index.values


# In[53]:

dataset = pd.merge(df, USTBonds, on='Date') #Daily
dataset = pd.merge(dataset,USDIndexFutures, on = 'Date') # Daily
#dataset = pd.merge(dataset,USInflationRate, on = 'Date') # Monthly
#dataset = pd.merge(dataset,USUnemp, on = 'Date') # Monthly
dataset = pd.merge(dataset,USFedFundRate, on = 'Date') # Daily
#dataset = pd.merge(dataset,USGDP, on = 'Date') # Annually


# In[54]:

dataset


# In[58]:

import plotly.plotly as py
import plotly.graph_objs as go



datasetPlot = [go.Scatter( x=dataset['Date'], y=dataset['close'] )]

py.iplot(datasetPlot, filename='pandas-time-series')

