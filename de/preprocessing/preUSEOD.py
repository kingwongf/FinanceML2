import pandas as pd
import os
print(os.getcwd())
os.chdir("/Users/kingf.wong/Development/FinanceML2")
import numpy as np
from tools import featGen
from scipy.stats.mstats import zscore, winsorize
import swifter
import yaml
from de.feat_engineering.technicalsGen import featTAGen
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 500)



class preUSEOD(object):
	def __init__(self, us_eod_loc, adj_close_loc, ticker_li_loc):
		self.us_eod_loc = us_eod_loc
		self.adj_close_loc = adj_close_loc
		self.ticker_li_loc = ticker_li_loc

	def pivot(self, values):
		us_eod = pd.read_pickle(self.us_eod_loc)
		us_eod = us_eod[us_eod['ticker'].isin(pd.read_csv(self.ticker_li_loc)['ticker'].tolist())]
		us_eod.index = pd.to_datetime(us_eod.Date)
		us_eod = us_eod.sort_index()
		us_eod = us_eod[[values, 'ticker']]
		us_eod = pd.pivot_table(us_eod, values=values, index=[us_eod.index],
		                                  columns=['ticker']).sort_index().fillna(method='ffill')
		# us_eod = us_eod[us_eod.columns[0:460]]
		us_eod.columns = pd.MultiIndex.from_product([['price'], us_eod.columns.tolist()], names=['feat', 'ticker'])
		return us_eod
	@staticmethod
	def flatten(df):
		return df.reset_index(level=0)







'''
us_eod = pd.read_pickle('data/US_EOD_20191019.pkl')

us_eod.index = pd.to_datetime(us_eod.Date)
us_eod = us_eod.sort_index()

us_eod_Adj_Close = us_eod[['Adj_Close', 'ticker']]

us_eod_Adj_Close = pd.pivot_table(us_eod, values='Adj_Close', index=[us_eod_Adj_Close.index],columns=['ticker'])
us_eod_Adj_Close = us_eod_Adj_Close.sort_index().fillna(method='ffill')
us_eod_Adj_Close = us_eod_Adj_Close['2006-01-01':'2019-10-18']

us_eod_Adj_Close.to_pickle('data/equities/preprocessed/us_eod_adj_close.pkl')


us_eod_Adj_Volume = us_eod[['Adj_Volume', 'ticker']]

us_eod_Adj_Volume = pd.pivot_table(us_eod, values='Adj_Volume', index=[us_eod_Adj_Volume.index],columns=['ticker'])
us_eod_Adj_Volume = us_eod_Adj_Volume.sort_index().fillna(method='ffill')
us_eod_Adj_Volume = us_eod_Adj_Volume['2008-01-01':'2019-10-18']

us_eod_Adj_Volume.to_pickle('data/equities/preprocessed/us_eod_adj_volume.pkl')

'''
