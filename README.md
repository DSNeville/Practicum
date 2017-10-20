# Practicum
### Regis University Data Science Practicum

#### John Neville

### Objective

Evaluate a series of different models to determine if we can come up with a viable predictive model for the cryptocurrency Ethereum.

###### Alternate Objective
 
Learn how to perform many of the data science techniques that I have done in R, but in the Python environment.

### Research


### Data 

Data is retrieved via several API calls.
We have called data from Quandl and Cryptocompare, both of which are free for the purposes of this exercise.
The datasets that I have called at this point are as follows:

*Ethereum pricing
*US Treasury Bond Values
*US Futures Index
*US Inflation Rate
*US Unemployment
*US Federal Fund Rate (Interest)
*S&P 500
*BTC to USD pricing
*GDP
*Lagged Values from above

The code to compile the data is below:

```

import json
import requests
import quandl
import pandas as pd
import datetime

from pandas.io.json import json_normalize

# Set API Key
quandl.ApiConfig.api_key = "F5A8zK7CGH9z8B9f_Brf"

# Call ETH Data
response = requests.get("https://min-api.cryptocompare.com/data/histoday?fsym=ETH&tsym=USD&allData=TRUE")

# Create Dataframe
data = json.loads(response.text)
data2 = json_normalize(data)
json_data = data['Data']
df=pd.DataFrame(data['Data'])
df['Date'] = pd.to_datetime(df['time'],unit='s')
df.dtypes
df.astype(object)

Eth = df[['Date','open','high','low','volumeto','volumefrom']]
Eth['ETHOpen'] = Eth['open']
Eth['ETHHigh'] = Eth['high']
Eth['ETHLow'] = Eth['low']
Eth = Eth[['Date','ETHOpen','ETHHigh','ETHLow','volumeto','volumefrom']]
Eth['Date'] = pd.to_datetime(Eth['Date']).dt.date

# US Treasury Bond Futures
USTBond = quandl.get('CHRIS/CME_US1')

# US Dollar Index Futures
USDIndexFuture = quandl.get('CHRIS/ICE_DX2')
    
# Inflation YOY - USA
USInflation = quandl.get('RATEINF/INFLATION_USA')

# Unemployment Level
USUn = quandl.get('FRED/UNRATE')
    
# Effective Federal Funds Rate (Interest Rate)
USFedFund = quandl.get('FRED/DFF')

# United States GDP at Current Prices, LCU Billions
GDP = quandl.get('ODA/USA_NGDP')

# S&P 500
SANDP = quandl.get('CHRIS/CME_SP1')



#Set values to expand data set and merge by date

USTBonds = pd.DataFrame(USTBond)
USTBonds['Date'] = USTBonds.index.values
USTBonds['Date'] = pd.to_datetime(USTBonds['Date']).dt.date
USTBonds.insert(0, 'NumInd', range(0, 0 + len(USTBonds)))
USTBonds.set_index('NumInd', inplace=True)
USTBonds['Value'] = USTBonds['Open']
USTBonds = USTBonds[['Date','Value']]
USTBonds['start'] = USTBonds['Date']
USTBonds['end'] = USTBonds['Date']
USTBonds.end = USTBonds.end.shift(-1)
USTBonds.end = USTBonds['end'].fillna(datetime.date.today())

USDIndexFutures = pd.DataFrame(USDIndexFuture)
USDIndexFutures['Date'] = USDIndexFutures.index.values
USDIndexFutures['Date'] = pd.to_datetime(USDIndexFutures['Date']).dt.date
USDIndexFutures.insert(0, 'NumInd', range(0, 0 + len(USDIndexFutures)))
USDIndexFutures.set_index('NumInd', inplace=True)
USDIndexFutures['Value'] = USDIndexFutures['Open']
USDIndexFutures = USDIndexFutures[['Date','Value']]
USDIndexFutures['start'] = USDIndexFutures['Date']
USDIndexFutures['end'] = USDIndexFutures['Date']
USDIndexFutures.end = USDIndexFutures.end.shift(-1)
USDIndexFutures.end = USDIndexFutures['end'].fillna(datetime.date.today())

USFedFundRate = pd.DataFrame(USFedFund)
USFedFundRate['Date'] = USFedFundRate.index.values
FedFundRate = USFedFundRate[['Date','Value']]
FedFundRate['FedFundRateValue'] = FedFundRate['Value']
FedFundRate = FedFundRate[['Date','FedFundRateValue']]
FedFundRate['Date'] = pd.to_datetime(FedFundRate['Date']).dt.date


SandP500 = pd.DataFrame(SANDP)
SandP500['Date'] = SandP500.index.values
SandP500['Date'] = pd.to_datetime(SandP500['Date']).dt.date
SandP500.loc[SandP500['Date'] == '2017-09-15', 'Value'] = '2492.50'
SandP500.insert(0, 'NumInd', range(0, 0 + len(SandP500)))
SandP500.set_index('NumInd', inplace=True)
SandP500['Value'] = SandP500['Open']
SandP500 = SandP500[['Date','Value']]
SandP500['start'] = SandP500['Date']
SandP500['end'] = SandP500['Date']
SandP500.end = SandP500.end.shift(-1)
SandP500.end = SandP500['end'].fillna(datetime.date.today())

USInflationRate = pd.DataFrame(USInflation)
USInflationRate['Date'] = USInflationRate.index.values
USInflationRate['Date'] = pd.to_datetime(USInflationRate['Date']).dt.date
USInflationRate.insert(0, 'NumInd', range(0, 0 + len(USInflationRate)))
USInflationRate.set_index('NumInd', inplace=True)
USInflationRate['start'] = USInflationRate['Date']
USInflationRate['end'] = USInflationRate['Date']
USInflationRate.end = USInflationRate.end.shift(-1)
USInflationRate.end = USInflationRate['end'].fillna(datetime.date.today())

USUnemp = pd.DataFrame(USUn)
USUnemp['Date'] = USUnemp.index.values
USUnemp['Date'] = pd.to_datetime(USUnemp['Date']).dt.date
USUnemp.insert(0, 'NumInd', range(0, 0 + len(USUnemp)))
USUnemp.set_index('NumInd', inplace=True)
USUnemp['start'] = USUnemp['Date']
USUnemp['end'] = USUnemp['Date']
USUnemp.end = USUnemp.end.shift(-1)
USUnemp.end = USUnemp['end'].fillna(datetime.date.today())

USGDP = pd.DataFrame(GDP)
USGDP['Date'] = USGDP.index.values
USGDP['Date'] = pd.to_datetime(USGDP['Date']).dt.date
USGDP.insert(0, 'NumInd', range(0, 0 + len(USGDP)))
USGDP.set_index('NumInd', inplace=True)
USGDP['start'] = USGDP['Date']
USGDP['end'] = USGDP['Date']
USGDP.end = USGDP.end.shift(-1)
USGDP.end = USGDP['end'].fillna(datetime.date.today())

#Loop through each subset of data and expand values that are missing between days

def full_data(dataframe):
    allframes = []
    for i in dataframe.index:
        newframe = pd.DataFrame()
        newframe['Date'] = pd.date_range(dataframe.iloc[i].start, dataframe.iloc[i].end, freq = 'D')
        newframe['value'] = dataframe.iloc[i]['Value']    
        allframes.append(newframe)
    return pd.concat(allframes)


USTBonds = full_data(USTBonds)
USTBonds = USTBonds[['Date','value']]
USTBonds['TBondsOpenValue'] = USTBonds['value']
USTBonds = USTBonds[['Date','TBondsOpenValue']]
USTBonds.drop(0,inplace=True)
USTBonds['Date'] = pd.to_datetime(USTBonds['Date']).dt.date

USDIndexFutures = full_data(USDIndexFutures)
USDIndexFutures = USDIndexFutures[['Date','value']]
USDIndexFutures['IndexFutures'] = USDIndexFutures['value']
USDIndexFutures = USDIndexFutures[['Date','IndexFutures']]
USDIndexFutures.drop(0,inplace=True)
USDIndexFutures['Date'] = pd.to_datetime(USDIndexFutures['Date']).dt.date

SandP500 = full_data(SandP500)
SandP500 = SandP500[['Date','value']]
SandP500['SandPValue'] = SandP500['value']
SandP500 = SandP500[['Date','SandPValue']]
SandP500.drop(0,inplace=True)
SandP500['Date'] = pd.to_datetime(SandP500['Date']).dt.date

InflationCast = full_data(USInflationRate)
InflationRate = InflationCast[['Date','value']]
InflationRate['InflationRateValue'] = InflationRate['value']
InflationRate = InflationRate[['Date','InflationRateValue']]
InflationRate['Date'] = pd.to_datetime(InflationRate['Date']).dt.date
InflationRate.drop(0,inplace=True)


UnempCast = full_data(USUnemp)
UnempRate = UnempCast[['Date','value']]
UnempRate['UnemploymentValue'] = UnempRate['value']
UnempRate = UnempRate[['Date','UnemploymentValue']]
UnempRate['Date'] = pd.to_datetime(UnempRate['Date']).dt.date
UnempRate.drop(0,inplace=True)

GDPCast = full_data(USGDP)
GDP = GDPCast[['Date','value']]
GDP['GDP'] = GDP['value']
GDP = GDP[['Date','GDP']]
GDP['Date'] = pd.to_datetime(GDP['Date']).dt.date
GDP.drop(0,inplace=True)

import plotly.plotly as py
from plotly.graph_objs import *

#Merge all sets

dataset = pd.merge(Eth, USTBonds,on='Date',how='left') #Daily
dataset = pd.merge(dataset,USDIndexFutures, on = 'Date', how='left') # Daily
dataset = pd.merge(dataset,InflationRate, on = 'Date', how='left') # Monthly
dataset = pd.merge(dataset,UnempRate, on = 'Date', how='left') # Monthly
dataset = pd.merge(dataset,FedFundRate, on = 'Date', how='left') # Daily
dataset = pd.merge(dataset,GDP, on = 'Date', how='left') # Annually
dataset = pd.merge(dataset,SandP500, on = 'Date', how='left') # Daily

#Fill in strangely missing data with what it would be

dataset.loc[dataset['Date'] == datetime.date(2017,9,15), 'SandPValue'] = 2492.5
dataset.loc[dataset['Date'] == datetime.date(2017,9,16), 'SandPValue'] = 2492.5
dataset.loc[dataset['Date'] == datetime.date(2017,9,17), 'SandPValue'] = 2492.5
dataset.loc[dataset['Date'] == datetime.date(2017,9,18), 'SandPValue'] = 2492.5

df = dataset

```


### Interpretation

In the data gathering document you can see that the data needed to be organized in a way to be analyzed.
The data is merged on a date field.  For measures that occur on a monthly basis, I cast them to repeat over missing days.

I begin with a few plots, but quickly realize that the scale of some of these measures is vastly different.  I scale some down by factors of ten to thousands, just to see their relative movement over time.

This can be viewed in the  "Load+All+Data" notebook.
In a section a little further along when we are looking for relationships in the data, I decided to add lagged values into the actual data set. 

### Exploration

I began by simply running correlation on my data, and making some plots.

Plot of original data set with the first features that I evaluate:

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/All%20Plot%20Explore.PNG)

Looking at the measure, I want to see how our values are distributed:

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/DistLogPrice.PNG)

A heatmap of the correlation values:

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/Heatmap%20Cor.PNG)

From here, I knew that I would need to add more features, particularly lagged values of some of the features I already had.
I add BTC information and lagged values, and plot another heatmap of the correlation.

```
dataset['ETHOpenTM1'] = dataset['ETHOpen'].shift(1)
dataset['ETHOpenTM2'] = dataset['ETHOpen'].shift(2)
dataset['ETHOpenTM3'] = dataset['ETHOpen'].shift(3)
dataset['ETHOpenTM4'] = dataset['ETHOpen'].shift(4)
dataset['ETHOpenTM5'] = dataset['ETHOpen'].shift(5)
dataset['ETHOpenTM6'] = dataset['ETHOpen'].shift(6)
dataset['ETHOpenTM7'] = dataset['ETHOpen'].shift(7)
dataset['BTCOpenTM1'] = dataset['BTCOpen'].shift(1)
dataset['BTCOpenTM2'] = dataset['BTCOpen'].shift(2)
dataset['BTCOpenTM3'] = dataset['BTCOpen'].shift(3)
dataset['BTCOpenTM4'] = dataset['BTCOpen'].shift(4)
dataset['BTCOpenTM5'] = dataset['BTCOpen'].shift(5)
dataset['BTCOpenTM6'] = dataset['BTCOpen'].shift(6)
dataset['BTCOpenTM7'] = dataset['BTCOpen'].shift(7)
dataset['PrevHigh'] = dataset['ETHHigh'].shift(1)
dataset['PrevHigh2'] = dataset['ETHHigh'].shift(2)
dataset['PrevHigh3'] = dataset['ETHHigh'].shift(3)
dataset['PrevHigh4'] = dataset['ETHHigh'].shift(4)
dataset['PrevLow'] = dataset['ETHLow'].shift(1)
dataset['PrevLow2'] = dataset['ETHLow'].shift(2)
dataset['PrevLow3'] = dataset['ETHLow'].shift(3)
dataset['PrevLow4'] = dataset['ETHLow'].shift(4)
dataset['PrevVolTo'] = dataset['volumeto'].shift(1)
dataset['PrevVolTo2'] = dataset['volumeto'].shift(2)
dataset['PrevVolTo3'] = dataset['volumeto'].shift(3)
dataset['PrevVolTo4'] = dataset['volumeto'].shift(4)
dataset['PrevVolFrom'] = dataset['volumefrom'].shift(1)
dataset['PrevVolFrom2'] = dataset['volumefrom'].shift(2)
dataset['PrevVolFrom3'] = dataset['volumefrom'].shift(3)
dataset['PrevVolFrom4'] = dataset['volumefrom'].shift(4)
dataset['PrevSP'] = dataset['SandPValue'].shift(1)
dataset['PrevSP2'] = dataset['SandPValue'].shift(2)
dataset['PrevSP3'] = dataset['SandPValue'].shift(3)
dataset['PrevSP4'] = dataset['SandPValue'].shift(4)
```




![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/All%20Models.PNG)

https://github.com/DSNeville/Practicum/blob/master/Images/All%20Models.PNG
