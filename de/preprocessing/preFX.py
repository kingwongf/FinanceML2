import numpy as np
import sklearn.covariance
import datetime
from datetime import date
import os
from functools import reduce
import pandas as pd
from time import process_time
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import zipfile
import time
from de.feat_gen.technicalsGen import featTAGen

from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


''' merge dataframes of different dates to have one large dataframe '''

class preFX(object):
	def __init__(self, tickers, tick_loc, all_loc, sample_freq='D'):
		self.tickers = tickers
		self.tick_loc = tick_loc
		self.all_loc = all_loc
		self.sample_freq = sample_freq
		if self.sample_freq == 'D':
			self.pickle_loc = "/daily_fx.pkl"
		else:
			self.pickle_loc = "/intra_" + self.sample_freq + ".pkl"

	def gen(self):
		for root, _, files in os.walk(self.tick_loc):
			print(root)
			dfs = []
			if '.DS_Store' in files: files.remove('.DS_Store')
			if self.pickle_loc[1:] in files: continue
			if len(files) != 0:
				for file in files:
					ticker = file[:6]
					if ticker in tickers:
						with zipfile.ZipFile(root + "/" + file, "r") as zip_ref:
							zip_ref.extractall(root)
						csv_path = root + "/" + file[:-4] + ".csv"
						df = pd.read_csv(csv_path, header=None,
						                 names=['ticker', 'date', ticker + '_bid', ticker + '_ask'], low_memory=False)
						df['mid_price'] = (df[ticker + '_bid'] + df[ticker + '_ask'])/2
						os.remove(csv_path)
						df.index = pd.to_datetime(df.date)
						df = df.drop(['ticker', 'date', ticker + '_bid', ticker + '_ask'], axis=1)
						dfs.append(df)
				# print(len(dfs))

				dfs_ = reduce(lambda X, x: pd.merge_asof(X[X.index.notna()].sort_index(), x[x.index.notna()].sort_index(),
				                                         left_index=True, right_index=True, direction='forward',
				                                         tolerance=pd.Timedelta('2ms')), dfs)
				dfs_ = dfs_.resample(self.sample_freq).first()
				# print(dfs_.columns)
				dfs_.to_pickle(root + self.pickle_loc)

	def merge(self):
		dfs = []
		for root, _, files in os.walk(self.tick_loc):
			if '.DS_Store' in files: files.remove('.DS_Store')
			if len(files) != 0:
				for file in files:
					if file == "daily_fx.pkl":
						df = pd.read_pickle(root + self.pickle_loc)
						df.columns = pd.MultiIndex.from_product([['price'], df.columns.tolist()], names=['feat', 'ticker'])
						dfs.append(df)
		dfs_ = reduce(lambda X, x: X.sort_index().append(x.sort_index()), dfs)
		dfs_ = dfs_.loc[~dfs_.index.duplicated(keep='first')].sort_index().dropna(axis=0).astype('float64')
		if self.all_loc == "data/fx/daily":
			dfs_.to_pickle(self.all_loc + "/all_fx_daily.pkl")
		else:
			dfs_.to_pickle(self.all_loc + "/all_fx_intra_%s.pkl" % self.sample_freq)




tickers = ['AUDJPY', 'AUDNZD', 'AUDUSD'
            , 'CADJPY', 'CHFJPY', 'EURCHF'
            , 'EURGBP', 'EURJPY', 'EURUSD'
            , 'GBPJPY', 'GBPUSD', 'NZDUSD'
            , 'USDCAD', 'USDCHF', 'USDJPY']
tick_loc = "data/fx/tick"
daily_loc = "data/fx/daily"

t = time.process_time()
tick_source = preFX(tickers, tick_loc, daily_loc, 'min')
tick_source.gen()
# tick_source.merge()

ta = featTAGen("data/fx/daily/all_fx_daily.pkl", '1m')
df = ta.ta_gen()
df.to_pickle('pre_data/feat_fx_daily.pkl')


# fx_loc = "data/fx/tick"
# today = date.today()




'''
for root, _, files in os.walk(fx_loc):
    dfs = []
    if '.DS_Store' in files: files.remove('.DS_Store')
    if len(files)!=0:
        for file in files:
            ticker = file[:6]
            if ticker in tickers:
                with zipfile.ZipFile(root + "/" + file, "r") as zip_ref:
                    zip_ref.extractall(root)
                csv_path = root + "/" + file[:-4] + ".csv"
                df = pd.read_csv(csv_path, header=None, names=['ticker', 'date', ticker + '_bid', ticker + '_ask'], low_memory=False)
                os.remove(csv_path)
                df.index = pd.to_datetime(df.date)
                df = df.drop(['ticker', 'date'],axis=1)
                dfs.append(df)
        print(len(dfs))
        dfs_ = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(),
                                            left_index=True, right_index=True, direction='forward',
                                            tolerance=pd.Timedelta('2ms')), dfs)
        dfs_ = dfs_.resample('D').first()
        print(dfs_.columns)
        dfs_.to_pickle(root + "/daily_fx.pkl")
        
'''
elapsed_time = time.process_time() - t
print(elapsed_time, " s")

