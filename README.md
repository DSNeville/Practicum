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
*Additionally, I will soon be looking into a sentiment on related news stories.

### Interpretation

In the data gathering document you can see that the data needed to be organized in a way to be analyzed.
The data is merged on a date field.  For measures that occur on a monthly basis, I cast them to repeat over missing days.

I begin with a few plots, but quickly realize that the scale of some of these measures is vastly different.  I scale some down by factors of ten to thousands, just to see their relative movement over time.

This can be viewed in the  "Load+All+Data" notebook.

From this point, I need to teach myself how to use the capabilities of Python to evaluate data.
I start by going through the rythms of a time series analysis.  I used a walk through, upating outdated functions from Analytics Vidhya.
(https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/).  For the most part this is complete, however I need to dig deeping to get some of these methods to align.
The next step from here is to look at other time series techniques that include other factors.


To Do

http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
Feature Selection again!
Tune Model
Reverse Lag (Was Backwards)
Add TS guesses
Compare Error
Make Final Model for t+1
Organize Repo
Make Presentation



