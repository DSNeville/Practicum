
# coding: utf-8

# In[103]:

import json
import requests
import quandl
import pandas as pd
import datetime

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


# In[104]:



USTBonds = pd.DataFrame(USTBonds)
USTBonds['Date'] = USTBonds.index.values
TBonds = USTBonds[['Date','Open']]
TBonds = TBonds.rename(columns = {'Open':'TBondsOpenValue'})

USDIndexFutures = pd.DataFrame(USDIndexFutures)
USDIndexFutures['Date'] = USDIndexFutures.index.values
IndexFutures = USDIndexFutures[['Date','Open']]
IndexFutures = IndexFutures.rename(columns = {'Open','IndexFutures'})

USFedFundRate = pd.DataFrame(USFedFundRate)
USFedFundRate['Date'] = USFedFundRate.index.values
FedFundRate = USFedFundRate[['Date','Value']]
FedFundRate = FedFundRate.rename(columns = {'Open','FedFundRateValue'})


USInflationRate = pd.DataFrame(USInflationRate)
USInflationRate['Date'] = USInflationRate.index.values
USInflationRate.insert(0, 'NumInd', range(0, 0 + len(USInflationRate)))
USInflationRate.set_index('NumInd', inplace=True)
USInflationRate['start'] = USInflationRate['Date']
USInflationRate['end'] = USInflationRate['Date']
USInflationRate.end = USInflationRate.end.shift(-1)
USInflationRate.end = USInflationRate['end'].fillna(datetime.date.today())

USUnemp = pd.DataFrame(USUnemp)
USUnemp['Date'] = USUnemp.index.values
USUnemp.insert(0, 'NumInd', range(0, 0 + len(USUnemp)))
USUnemp.set_index('NumInd', inplace=True)
USUnemp['start'] = USUnemp['Date']
USUnemp['end'] = USUnemp['Date']
USUnemp.end = USUnemp.end.shift(-1)
USUnemp.end = USUnemp['end'].fillna(datetime.date.today())

USGDP = pd.DataFrame(USGDP)
USGDP['Date'] = USGDP.index.values
USGDP.insert(0, 'NumInd', range(0, 0 + len(USGDP)))
USGDP.set_index('NumInd', inplace=True)
USGDP['start'] = USGDP['Date']
USGDP['end'] = USGDP['Date']
USGDP.end = USGDP.end.shift(-1)
USGDP.end = USGDP['end'].fillna(datetime.date.today())


# In[87]:

def full_data(dataframe):
    allframes = []
    for i in dataframe.index:
        newframe = pd.DataFrame()
        newframe['Date'] = pd.date_range(dataframe.iloc[i].start, dataframe.iloc[i].end, freq = 'D')
        newframe['value'] = dataframe.iloc[i]['Value']    
        allframes.append(newframe)
    return pd.concat(allframes)


InflationCast = full_data(USInflationRate)
InflationRate = InflationCast[['Date','value']]
InflationRate = InflationRate.rename(columns = {'value','InflationRateValue'})

UnempCast = full_data(USUnemp)
UnempRate = UnempCast[['Date','value']]
UnempRate = UnempRate.rename(columns = {'value','UnemploymentValue'})

GDPCast = full_data(USGDP)
GDP = GDPCast[['Date','value']]
GDP = GDP.rename(columns = {'value','GDP'})


# In[88]:

dataset = pd.merge(df, USTBonds, on='Date') #Daily
dataset = pd.merge(dataset,USDIndexFutures, on = 'Date') # Daily
dataset = pd.merge(dataset,InflationRate, on = 'Date') # Monthly
dataset = pd.merge(dataset,UnempRate, on = 'Date') # Monthly
dataset = pd.merge(dataset,USFedFundRate, on = 'Date') # Daily
dataset = pd.merge(dataset,GDP, on = 'Date') # Annually


# In[89]:

dataset


# In[105]:




# In[18]:

py.iplot([{
    'x': dataset['Date'],
    'y': df[col],
    'name': col
}  for col in df.columns], filename='dataset')


# In[77]:

import plotly.plotly as py
from plotly.graph_objs import *
plotly.tools.set_credentials_file(username='metatortoise', api_key='3YfeZWJz0EALbtko5rGa')
plotly.tools.set_config_file(world_readable=True,
                             sharing='public')

import plotly.tools as tls
tls.embed('https://plot.ly/~cufflinks/8')


datasetPlot = [go.Scatter( x=dataset['Date'], y=dataset['close'] )]

py.iplot(datasetPlot, filename='pandas-time-series')


# In[107]:

USDIndexFutures = quandl.get('CHRIS/ICE_DX2')


# In[113]:

USDIndexFutures = pd.DataFrame(USDIndexFutures)
USDIndexFutures['Date'] = USDIndexFutures.index.values
USDIndexFutures 
IndexFutures = USDIndexFutures[['Date','Open']]
IndexFutures


# In[118]:

GDP = GDPCast[['Date','value']]
GDP = GDP.rename(columns = {'value','GDP'})

