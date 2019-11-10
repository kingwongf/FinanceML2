import pandas as pd
import numpy as np
from tools import featGen
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 500)
from scipy.stats.mstats import zscore, winsorize
import swifter
from functools import reduce


columns = pd.MultiIndex.from_product([['price'], ['ticker1', 'ticker2','ticker3']], names=['feat', 'ticker'])

df = pd.DataFrame(np.random.rand(10,3),columns=columns)

df.index = pd.to_datetime(df.index)
df.index.name = 'Date'
# print(df)

# print(df.info())
# print(df.apply(df.swifter.apply(featGen.momentum(1), axis=0)))
# print(df.swifter.apply(featGen.momentum, axis=0, args=(1, )))


#print(df.groupby(level=0, axis=1).apply(lambda x: x+1))
'''
df = df.join(df.apply(lambda x: x +1 ).rename(columns={"price": "add_1"}))\
      .join(df.swifter.apply(featGen.momentum, axis=0, args=(1, )).rename(columns={"price": "mom1"}))\
      .join(df.swifter.apply(featGen.momentum, axis=0, args=(3, )).rename(columns={"price": "mom3"}))\
      .join(df.swifter.apply(featGen.momentum, axis=0, args=(3, )).diff(periods=1).rename(columns={"price": "chmom3"}))
'''
feat_li = [(featGen.momentum, 1, "mom1"), (featGen.momentum, 3, "mom3"), (featGen.chmom, 3, "chmom3"), (featGen.ret, 5, "return1m"), (featGen.MACD, (1,2), "MACD112")]
for item in feat_li:
      if len(item) ==3:
            df = df.join(df[['price']].swifter.apply(item[0], axis=0, args=(item[1], )).fillna(method='ffill').rename(columns={"price": item[2]}))


# print(df)
print(df.unstack('feat').unstack('feat'))

name_dfs = ['close', 'mom1d', 'mom1w', 'mom1m', 'chmom1m', 'mom6m', 'chmom6m', 'mom12m',
            'chmom12m', 'retvol1m', 'retvol12m', 'maxret1m',
            'maxret12m', 'ema1m', 'RSI', 'MACD1m12m', 'return1m', 'emaret1m', 'fwd_return1m']

freq_dict = {'1d': 1,
                  '1w': 5,
                  '1m': 20,
                  '6m': 125,
                  '12m': 250
                  }
feat_dict = {'mom': featGen.momentum,
             'retvol': featGen.retvol,
             'ema': featGen.ema,
             'RSI': featGen.RSI,
             'MACD': featGen.MACD,
             'maxret': featGen.maxret
             }

RSI = us_eod_Adj_Close.apply(featGen.RSI, axis=0).fillna(method='ffill').unstack().reset_index(name='RSI')
MACD1m12m = us_eod_Adj_Close.apply(featGen.MACD, axis=0).fillna(method='ffill').unstack().reset_index(name='MACD1m12m')

return1m = us_eod_Adj_Close.pct_change(20).unstack().reset_index(name='return1m')
emaret1m = us_eod_Adj_Close.pct_change(20).rolling(20).mean().unstack().reset_index(name='emaret1m')
fwd_return1m = us_eod_Adj_Close.pct_change(20).shift(-20).unstack().reset_index(name='fwd_return1m')


feat_li = [(featGen.momentum, "mom1d", freq_dict['1d']),
           (featGen.momentum, "mom1w", freq_dict['1w']),
           (featGen.momentum, "mom1m", freq_dict['1m']),
           (featGen.momentum, "mom6m", freq_dict['6m']),
           (featGen.momentum, "mom12m", freq_dict['12m']),
           (featGen.retvol, "retvol1m", freq_dict['1m']),
           (featGen.retvol, "retvol12m", freq_dict['12m']),
           (featGen.maxret, "retvol1m", freq_dict['1m']),
           (featGen.maxret, "retvol12m", freq_dict['12m']),
           (featGen.ema, "ema1m", freq_dict['1m']),
           (featGen.RSI, "RSI", 14),
           (featGen.MACD, "MACD", (20, 250)),
           (featGen.chmom , "chmom1m", freq_dict['1m']),
           (featGen.chmom, "chmom6m", freq_dict['6m']),
           (featGen.chmom, "chmom12m", freq_dict['12m']),
           ]

def feat_apply(df, **kwargs):
      df = df.join(df[['price']].swifter.apply(item[0], axis=0, args=(item[1],)).fillna(method='ffill').rename(
            columns={"price": item[2]}))


# df1.unstack(level=0)
# df1 = df.assign(lambda x: x.groupby(level=0, axis=1) + 1)
# df.assign(add1=lambda x: x.price + 1)

# print(df)