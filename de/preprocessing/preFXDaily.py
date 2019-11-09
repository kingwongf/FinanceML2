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


from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


''' merge dataframes of different dates to have one large dataframe '''

class preFXDaily(object):
	def __init__(self, tickers, tick_loc, daily_loc):
		self.tickers = tickers
		self.tick_loc = tick_loc
		self.daily_loc = daily_loc
	def gen_daily(self):
		for root, _, files in os.walk(self.tick_loc):
			print(root)
			dfs = []
			if '.DS_Store' in files: files.remove('.DS_Store')
			if 'daily_fx.pkl' in files: continue
			if len(files) != 0:
				for file in files:
					ticker = file[:6]
					if ticker in tickers:
						with zipfile.ZipFile(root + "/" + file, "r") as zip_ref:
							zip_ref.extractall(root)
						csv_path = root + "/" + file[:-4] + ".csv"
						df = pd.read_csv(csv_path, header=None,
						                 names=['ticker', 'date', ticker + '_bid', ticker + '_ask'], low_memory=False)
						os.remove(csv_path)
						df.index = pd.to_datetime(df.date)
						df = df.drop(['ticker', 'date'], axis=1)
						dfs.append(df)
				# print(len(dfs))
				dfs_ = reduce(lambda X, x: pd.merge_asof(X[pd.notna(X.index)].sort_index(), x[pd.notna(x.index)].sort_index(),
				                                         left_index=True, right_index=True, direction='forward',
				                                         tolerance=pd.Timedelta('2ms')), dfs)
				dfs_ = dfs_.resample('D').first()
				# print(dfs_.columns)
				dfs_.to_pickle(root + "/daily_fx.pkl")

	def merge_daily(self):
		dfs = []
		for root, _, files in os.walk(self.tick_loc):
			if '.DS_Store' in files: files.remove('.DS_Store')
			if len(files) != 0:
				for file in files:
					if file == "daily_fx.pkl":
						df = pd.read_pickle(root + "/daily_fx.pkl")
						df.index = df.date
						dfs.append(df)
		dfs_ = reduce(lambda X, x: X.sort_index().append(x.sort_index()), dfs)
		dfs_ = dfs_.loc[~dfs_.index.duplicated(keep='first')].sort_index().interpolate()
		dfs_.to_pickle(self.daily_loc + "/daily.pkl")




tickers = ['AUDJPY', 'AUDNZD', 'AUDUSD'
            , 'CADJPY', 'CHFJPY', 'EURCHF'
            , 'EURGBP', 'EURJPY', 'EURUSD'
            , 'GBPJPY', 'GBPUSD', 'NZDUSD'
            , 'USDCAD', 'USDCHF', 'USDJPY']
tick_loc = "data/fx/tick"
daily_loc = "data/fx/daily"

t = time.process_time()
tick_source = preFXDaily(tickers, tick_loc, daily_loc)
tick_source.gen_daily()
# tick_source.merge_daily()

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

