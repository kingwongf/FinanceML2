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
from de.preprocessing.preUSEOD import preUSEOD
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 500)



fwd_freq = '3d'
pre = preUSEOD('data/equities/US_EOD_20191019.pkl', 'data/equities/preprocessed/us_eod_adj_close.pkl', 'data/equities/revolut_tickers.csv')
df_pre_useod = pre.pivot('Adj_Close').fillna(method='ffill')
ta = featTAGen(df_pre_useod, fwd_freq)
df = ta.ta_gen()
df = pre.flatten(df).dropna(axis=0)['2009-01-01':]   ## debugging ['2018-01-01':'2018-01-20']
df.to_pickle('pre_data/feat_useod_daily_%sfwd.pkl' % fwd_freq)
