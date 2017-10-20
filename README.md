# Practicum
### Regis University Data Science Practicum

#### John Neville

### Objective

Evaluate a series of different models to determine if we can come up with a viable predictive model for the cryptocurrency Ethereum.

###### Alternate Objective
 
Learn how to perform many of the data science techniques that I have done in R, but in the Python environment.


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
This can be viewed in the  "Load+All+Data" notebook.

### Interpretation

In the data gathering document you can see that the data needed to be organized in a way to be analyzed.
The data is merged on a date field.  For measures that occur on a monthly basis, I cast them to repeat over missing days.

I begin with a few plots, but quickly realize that the scale of some of these measures is vastly different.  I scale some down by factors of ten to thousands, just to see their relative movement over time.

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/All%20Plot%20Explore.PNG)

In a section a little further along when we are looking for relationships in the data, I decided to add lagged values into the actual data set, but for now we can move onto the next segment.  The data is ready for interpretation.

### Exploration

I began by simply running correlation on my data, and making some plots.


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

With the lagged variables, we take a look at a another heatmap, forgive me as it is rather large.

```
corr = dataset.corr(method='pearson')
#corrmat = np.abs(corrmat)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(255, 10, as_cmap=True)

f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,xticklabels = True, yticklabels = True, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
 ```

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/Heatmap%20Large.PNG)

With some understanding, we can start looking at the models.

### Models:
 * Timeseries Arima
 * Timeseries FBProphet
 * Linear Regression
 * Ridge Regression
 * Lasso Regression
 * Mixed Models
 
#### Timeseries Models

In the timeseries evaluation, we looked at several things.  I eventually used ARIMA, but beforehand had tried using moving averages and
log values to achieve a stationary model.
I took advantage of the Dickey-Fuller test to see if I could achieve stationary results.  
Below is an example or one of the plots

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/TS%20Rolling%20Avg.PNG)

I then siwtched to ARIMA, using code from machinelearningmastery.com to optimize my ARIMA values.

```
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tsa.arima_model import ARIMA

def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
 
# load dataset
ts_log_moving_avg_diff = ts_log_moving_avg_diff[ts_log_moving_avg_diff.index>'2017-06-05']
series = ts_log_moving_avg_diff
# evaluate parameters
p_values = [0, 1, 2, 4]
d_values = range(0, 2)
q_values = range(0, 6)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

This gave me something to use for this model, but we will revisit it later.

The other timseries model I looked at was using FBProphet.

This is very simple to use:

```
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from fbprophet import Prophet

data = pd.read_csv('C:/Users/JP/Documents/School/Practicum/Github/Practicum/data/dataset.csv')
#df[['ds','y']] = pd.DataFrame(data[['Date','ETHOpen']])
#df['ds'] = pd.to_datetime(df['ds'],format='%Y-%m-%d')
df = pd.DataFrame(data[['Date','ETHOpen']])
df[['ds','y']] = df 
df = df.set_index('Date')
df = df[df.index>'2017-01-30']
df = df[df.index<'2017-10-05']



m = Prophet(weekly_seasonality = True,yearly_seasonality=True)
m.fit(df) 
future = m.make_future_dataframe(periods=50)

forecast=m.predict(future)
```

A plot for this model:

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/Prophet%20TS.PNG)

#### Linear Regression/Ridge Regression/Lasso Regression
I put all three of these together in my code for organizational purposes.  There are pages where I evaluate them individually, but
to stay focused, let's keep them together
I used xgb boost for linear regression, inspiried by another repository to give me a prediction using many features.
At this point, I run this several times, having used some feature selection to get desired results, but to my surprise the
original seemed to perform the best regardless.
This is also true for my Ridge and Lasso regression.

The code:

```
#Here we test how our models run when using the hand picked features based off the correlation matrix
# We also consider a few features that we think still have some impact

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Read in data
data = pd.read_csv('C:/Users/JP/Documents/School/Practicum/Github/Practicum/data/dataset2.csv')

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df = df.set_index('Date')
df=df[df.index>'2017-05-30']

# Create Train and selection features for model
train=df[df.index<'2017-10-05']

features = ['TBondsOpenValue', 'UnemploymentValue', 'BTCOpenTM1', 'BTCOpenTM2',\
        'BTCOpenTM3', 'BTCOpenTM4', 'BTCOpenTM5', 'BTCOpenTM6', 'BTCOpenTM7','GDP','SandPValue',\
            'ETHOpenTM1', 'ETHOpenTM2','PrevHigh','PrevHigh2','PrevHigh3','PrevHigh4','PrevLow','PrevLow2','PrevLow3',\
            'PrevLow4','PrevVolTo','PrevVolTo2','PrevVolTo3','PrevVolTo4','PrevVolFrom','PrevVolFrom2','PrevVolFrom3',\
            'PrevVolFrom4','PrevSP','PrevSP2','PrevSP3','PrevSP4',\
        'ETHOpenTM3', 'ETHOpenTM4', 'ETHOpenTM5', 'ETHOpenTM6', 'ETHOpenTM7']


train=train.dropna()

#Set xgb matrix
dtrain = xgb.DMatrix(train.loc[:, features].values, \
                     label = train.loc[:, 'ETHOpen'].values)
#Set xgb parameters
params = {}
params['booster']  = 'gbtree'
params['objective'] = 'reg:linear'
params['max_depth'] = 6
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8
params['silent'] = 1
params['eval_metric'] = 'rmse'
num_round = 50
eval_list  = [(dtrain,'train')]

train['Date'] = train.index.values

print('Training xgb model:')
bst = xgb.train(params, dtrain, num_round, eval_list)

print('Train Ridge Regression:')
lr = Ridge()
lr.fit(train.loc[:, features].values, \
       train.loc[:, 'ETHOpen'].values)

print('Training Lasso Regression:')
lassoreg = Lasso(alpha=.001,normalize=True, max_iter=1e7)
lassoreg.fit(train.loc[:, features].values,train.loc[:, 'ETHOpen'].values)
 
test1 = df[df.index>='2017-10-05']
test1=test1.dropna()

#Run Models and Forecast Values using Test values, target is ETHOpen
while True:
    dtest1 = xgb.DMatrix(test1[features].values)
    xgb_pred = bst.predict(dtest1)
    lr_pred = lr.predict(test1[features].values)
    lasso_pred = lassoreg.predict(test1.loc[:, features].values)
    test1['ETHOpenRidgexgb1'] = 0.2*xgb_pred+0.8*lr_pred
    test1['ETHOpenRidge1'] = lr_pred
    test1['ETHOpenxgb1'] = xgb_pred
    test1['ETHOpenLasso1'] = lasso_pred


    target = train['ETHOpen']
    
    done = 1
    
    if done:
        print("Prediction: {}".format(test1[['ETHOpen','ETHOpenRidgexgb1','ETHOpenRidge1','ETHOpenxgb1','ETHOpenLasso1']]))
        break
        
        
   
test1 = pd.DataFrame(test1[['ETHOpen','ETHOpenRidgexgb1','ETHOpenRidge1','ETHOpenxgb1','ETHOpenLasso1']])
```

![alt text](https://github.com/DSNeville/Practicum/blob/master/Images/All%20Models.PNG)

