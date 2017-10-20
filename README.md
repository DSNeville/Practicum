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
As of the time of this writing, we have called data from Quandl and Cryptocompare, both of which are free for the purposes of this exercise.
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

### Interpretation

In the data gathering document you can see that the data needed to be organized in a way to be analyzed.
The data is merged on a date field.  For measures that occur on a monthly basis, I cast them to repeat over missing days.

I begin with a few plots, but quickly realize that the scale of some of these measures is vastly different.  I scale some down by factors of ten to thousands, just to see their relative movement over time.

This can be viewed in the  "Load+All+Data" notebook.
In a section a little further along when we are looking for relationships in the data, I decided to add lagged values into the actual data set. 

### Exploration


