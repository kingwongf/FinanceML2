import pandas as pd
import numpy as np
from tools import featGen
from scipy.stats.mstats import zscore, winsorize
import swifter
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 500)

us_eod = pd.read_pickle('data/US_EOD_20191019.pkl')

us_eod['Date'] = pd.to_datetime(us_eod.Date)
us_eod.index = us_eod.Date
us_eod = us_eod.sort_index()

us_eod_Adj_Close = us_eod[['Adj_Close', 'ticker']]

us_eod_Adj_Close = pd.pivot_table(us_eod, values='Adj_Close', index=[us_eod_Adj_Close.index],columns=['ticker'])
us_eod_Adj_Close = us_eod_Adj_Close.sort_index().fillna(method='ffill')
us_eod_Adj_Close = us_eod_Adj_Close['2008-01-01':'2019-10-18']

us_eod_Adj_Close.to_pickle('data/equities/preprocessed/us_eod_adj_close.pkl')


us_eod_Adj_Volume = us_eod[['Adj_Volume', 'ticker']]

us_eod_Adj_Volume = pd.pivot_table(us_eod, values='Adj_Volume', index=[us_eod_Adj_Volume.index],columns=['ticker'])
us_eod_Adj_Volume = us_eod_Adj_Volume.sort_index().fillna(method='ffill')
us_eod_Adj_Volume = us_eod_Adj_Volume['2008-01-01':'2019-10-18']

us_eod_Adj_Volume.to_pickle('data/equities/preprocessed/us_eod_adj_volume.pkl')
