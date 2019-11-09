import pandas as pd
import numpy as np
from tools import featGen
from scipy.stats.mstats import zscore, winsorize
import swifter
from functools import reduce
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 500)



class featTAGen(object):
    def __init__(self, daily_price_loc, bid_ask_ind, ret_freq):
        self.df = pd.read_pickle(daily_price_loc).rename_axis(index='Date', columns="ticker")
        self.bid_ask_ind = bid_ask_ind
        self.ticker_names = self.df.columns.tolist()
        self.ret_freq = ret_freq
        self.feat_dict = {'mom': featGen.momentum,
                     'retvol': featGen.retvol,
                     'ema': featGen.ema,
                     'RSI': featGen.RSI,
                     'MACD': featGen.MACD,
                     'maxret': featGen.maxret
                     }
        self.freq_dict = {'1d': 1,
                     '1w': 5,
                     '1m': 20,
                     '6m': 125,
                     '12m': 250
                     }

    def df_ta(self, df, feat, freq):
        return df.swifter.apply(self.feat_dict[feat], axis=0, args=(self.freq_dict[freq],)).fillna(
            method='ffill').unstack().reset_index(name=feat + freq)

    def ta_gen(self):
        ret_freq = self.freq_dict[self.ret_freq]
        name_dfs = ['close','mom1d', 'mom1w', 'mom1m', 'chmom1m', 'mom6m', 'chmom6m', 'mom12m',
                    'chmom12m', 'retvol1m', 'retvol12m', 'maxret1m',
                    'maxret12m', 'ema1m', 'RSI', 'MACD1m12m', 'return1m', 'emaret1m', 'fwd_return1m']
        dfs = [self.df.unstack().reset_index(name='close'),
               self.df_ta('mom', '1d'), self.df_ta('mom', '1w'),
               self.df_ta('mom', '1m'), self.df_ta('mom', '1m').groupby('ticker')['mom1m'].diff(periods=1),
               self.df_ta('mom', '6m'), self.df_ta('mom', '6m').groupby('ticker')['mom6m'].diff(periods=1),
               self.df_ta('mom', '12m'), self.df_ta('mom', '12m').groupby('ticker')['mom12m'].diff(periods=1),
               self.df_ta('retvol', '1m'), self.df_ta('retvol', '12m'),
               self.df_ta('maxret', '1m'),
               self.df_ta('maxret', '12m'), self.df_ta('ema', '1m'),
               self.df.apply(featGen.RSI, axis=0).fillna(method='ffill').unstack().reset_index(name='RSI'),
               self.df.apply(featGen.MACD, axis=0).fillna(method='ffill').unstack().reset_index(name='MACD1m12m'),
               self.df.pct_change(ret_freq).unstack().reset_index(name='return' + self.ret_freq),
               self.df.pct_change(ret_freq).rolling(ret_freq).mean().unstack().reset_index(name='return' + self.ret_freq),
               self.df.pct_change(ret_freq).shift(-ret_freq).unstack().reset_index(name='fwd_return'+ self.ret_freq)]

        feat_dfs = zip(name_dfs, dfs)

        def mrm_c(std, vol):
            value = np.tanh((10 / vol) * 50 * std)
            value[value < -0.8] = -1
            value[value > 0.8] = 1
            value[(value >= -0.8) & (value <= 0.8)] = 0
            return value

        def get_norm_side(mean, vol, ret, z):
            side = winsorize(ret, limits=[0.025, 0.025])
            side[(ret - mean) / np.sqrt(vol) > z] = 1
            side[(ret - mean) / np.sqrt(vol) < -z] = -1
            side[((ret - mean) / np.sqrt(vol) >= -z) & ((ret - mean) / np.sqrt(vol) <= z)] = 0
            return side

        unstack_df = reduce(
            lambda X, x: pd.merge(X, x, how='left', left_on=['ticker', 'Date'], right_on=['ticker', 'Date']), dfs)

        unstack_df.columns = name_dfs
        unstack_df.index = pd.to_datetime(unstack_df.Date)

        unstack_df['side'] = get_norm_side(unstack_df.emaret1m, unstack_df.retvol1m, unstack_df.fwd_return1m, 1.645)


        self.df[['mom1d' + ticker for ticker in self.ticker_names]] = self.df_ta('mom', '1d')
        mom1w = self.df_ta('mom', '1w')
        mom1m = self.df_ta('mom', '1m')
        mom6m = self.df_ta('mom', '6m')
        mom12m = self.df_ta('mom', '12m')

        mom1m['chmom1m'] = mom1m.groupby('ticker')['mom1m'].diff(periods=1)
        mom6m['chmom6m'] = mom6m.groupby('ticker')['mom6m'].diff(periods=1)
        mom12m['chmom12m'] = mom12m.groupby('ticker')['mom12m'].diff(periods=1)

        retvol1m = self.df_ta('retvol', '1m')
        retvol12m = self.df_ta('retvol', '12m')
        maxret1m = self.df_ta('maxret', '1m')
        maxret12m = self.df_ta('maxret', '12m')

        ema1m = self.df_ta('ema', '1m')

        RSI = self.df.apply(featGen.RSI, axis=0).fillna(method='ffill').unstack().reset_index(name='RSI')
        MACD1m12m = self.df.apply(featGen.MACD, axis=0).fillna(method='ffill').unstack().reset_index(
            name='MACD1m12m')

        return1m = self.df.pct_change(20).unstack().reset_index(name='return1m')
        emaret1m = self.df.pct_change(20).rolling(20).mean().unstack().reset_index(name='emaret1m')
        fwd_return1m = self.df.pct_change(20).shift(-20).unstack().reset_index(name='fwd_return1m')

        def mrm_c(std, vol):
            value = np.tanh((10 / vol) * 50 * std)
            value[value < -0.8] = -1
            value[value > 0.8] = 1
            value[(value >= -0.8) & (value <= 0.8)] = 0
            return value

        def get_norm_side(mean, vol, ret, z):
            side = winsorize(ret, limits=[0.025, 0.025])
            side[(ret - mean) / np.sqrt(vol) > z] = 1
            side[(ret - mean) / np.sqrt(vol) < -z] = -1
            side[((ret - mean) / np.sqrt(vol) >= -z) & ((ret - mean) / np.sqrt(vol) <= z)] = 0
            return side

        dfs = [self.df, mom1d, mom1w, mom1m, mom6m, mom12m, retvol1m, retvol12m, maxret1m,
               maxret12m, ema1m, RSI, MACD1m12m, return1m, emaret1m, fwd_return1m]

        name_dfs = ['ticker', 'Date', 'adj_close', 'mom1d', 'mom1w', 'mom1m', 'chmom1m', 'mom6m', 'chmom6m', 'mom12m',
                    'chmom12m', 'retvol1m', 'retvol12m', 'maxret1m',
                    'maxret12m', 'ema1m', 'RSI', 'MACD1m12m', 'return1m', 'emaret1m', 'fwd_return1m']

        unstack_df = reduce(
            lambda X, x: pd.merge(X, x, how='left', left_on=['ticker', 'Date'], right_on=['ticker', 'Date']), dfs)

        unstack_df.columns = name_dfs
        unstack_df.index = pd.to_datetime(unstack_df.Date)

        unstack_df['side'] = get_norm_side(unstack_df.emaret1m, unstack_df.retvol1m, unstack_df.fwd_return1m, 1.645)

        print(unstack_df[['ticker', 'ema1m', 'retvol1m', 'fwd_return1m', 'side']])
        unstack_df.to_csv('data/unstack_us_eod_with_feat.csv')


ta = featTAGen("data/fx/tick/2009-05/daily_fx.pkl", True)
print(ta.df.head(5))

# ta.ta_gen()


'''
us_eod_Adj_Close = pd.read_pickle('data/us_eod_adj_close.pkl')
# us_eod_Adj_Volume = pd.read_pickle('data/us_eod_adj_volume.pkl')

feat_dict = {'mom': featGen.momentum,
             'retvol': featGen.retvol,
             'ema': featGen.ema,
             'RSI': featGen.RSI,
             'MACD': featGen.MACD,
             'maxret': featGen.maxret
             }
freq_dict = {'1d': 1,
             '1w': 5,
             '1m': 20,
             '6m': 125,
             '12m': 250
             }

def daily_us_eod_price_feat(feat, freq, unstack=True):
    if unstack:
        return us_eod_Adj_Close.swifter.apply(feat_dict[feat], axis=0, args=(freq_dict[freq], )).fillna(method='ffill').unstack().reset_index(name=feat+freq)
    else:
        return us_eod_Adj_Close.swifter.apply(feat_dict[feat], axis=0, args=(freq_dict[freq],)).fillna(method='ffill')

mom1d = daily_us_eod_price_feat('mom','1d')
mom1w = daily_us_eod_price_feat('mom','1w')
mom1m = daily_us_eod_price_feat('mom','1m')
mom6m = daily_us_eod_price_feat('mom','6m')
mom12m = daily_us_eod_price_feat('mom','12m')

mom1m['chmom1m'] = mom1m.groupby('ticker')['mom1m'].diff(periods=1)
mom6m['chmom6m'] = mom6m.groupby('ticker')['mom6m'].diff(periods=1)
mom12m['chmom12m'] = mom12m.groupby('ticker')['mom12m'].diff(periods=1)


retvol1m = daily_us_eod_price_feat('retvol','1m')
retvol12m = daily_us_eod_price_feat('retvol','12m')
maxret1m = daily_us_eod_price_feat('maxret','1m')
maxret12m = daily_us_eod_price_feat('maxret','12m')

ema1m = daily_us_eod_price_feat('ema','1m')


RSI = us_eod_Adj_Close.apply(featGen.RSI, axis=0).fillna(method='ffill').unstack().reset_index(name='RSI')
MACD1m12m = us_eod_Adj_Close.apply(featGen.MACD, axis=0).fillna(method='ffill').unstack().reset_index(name='MACD1m12m')

return1m = us_eod_Adj_Close.pct_change(20).unstack().reset_index(name='return1m')
emaret1m = us_eod_Adj_Close.pct_change(20).rolling(20).mean().unstack().reset_index(name='emaret1m')
fwd_return1m = us_eod_Adj_Close.pct_change(20).shift(-20).unstack().reset_index(name='fwd_return1m')


def mrm_c(std, vol):
    value = np.tanh((10 / vol) * 50 * std)
    value[value < -0.8] = -1
    value[value > 0.8] = 1
    value[(value >= -0.8) & (value <= 0.8)] = 0
    return value

def get_norm_side(mean, vol, ret, z ):
    side = winsorize(ret, limits = [0.025,0.025])
    side[(ret - mean)/ np.sqrt(vol) > z] = 1
    side[(ret - mean)/ np.sqrt(vol) < -z] = -1
    side[((ret - mean)/ np.sqrt(vol) >= -z) & ((ret - mean)/ np.sqrt(vol) <= z)] = 0
    return side

unstack_adj_close = us_eod_Adj_Close.unstack().reset_index(name='adj_close')

# side = fwd_return1m.apply(lambda x: get_norm_side(x.ema1m, x.retvol1m, x.fwd_return, 1.645), axis=1)
dfs = [unstack_adj_close, mom1d, mom1w, mom1m, mom6m, mom12m, retvol1m, retvol12m, maxret1m,
       maxret12m, ema1m, RSI, MACD1m12m, return1m, emaret1m, fwd_return1m]


name_dfs = ['ticker', 'Date', 'adj_close', 'mom1d', 'mom1w', 'mom1m', 'chmom1m', 'mom6m', 'chmom6m', 'mom12m', 'chmom12m', 'retvol1m', 'retvol12m', 'maxret1m',
       'maxret12m', 'ema1m', 'RSI', 'MACD1m12m', 'return1m', 'emaret1m', 'fwd_return1m']

unstack_df = reduce(lambda X, x: pd.merge(X, x,  how='left', left_on=['ticker','Date'], right_on = ['ticker','Date']) ,dfs)

unstack_df.columns = name_dfs
unstack_df.index = pd.to_datetime(unstack_df.Date)

unstack_df['side'] = get_norm_side(unstack_df.emaret1m, unstack_df.retvol1m, unstack_df.fwd_return1m, 1.645)

print(unstack_df[['ticker','ema1m', 'retvol1m', 'fwd_return1m','side']])
unstack_df.to_csv('data/unstack_us_eod_with_feat.csv')

'''
'''

feat_dict = {'mom': featGen.momentum,
             'retvol': featGen.retvol,
             'ema': featGen.ema,
             'RSI': featGen.RSI,
             'MACD': featGen.MACD,
             'maxret': featGen.maxret
             }
freq_dict = {'1d': 1,
             '1w': 5,
             '1m': 20,
             '6m': 125,
             '12m': 250
             }

def daily_us_eod_price_feat(feat, freq, unstack=True):
    if unstack:
        return us_eod_Adj_Close.swifter.apply(feat_dict[feat], axis=0, args=(freq_dict[freq], )).fillna(method='ffill').unstack().reset_index(name=feat+freq)
    else:
        return us_eod_Adj_Close.swifter.apply(feat_dict[feat], axis=0, args=(freq_dict[freq],)).fillna(method='ffill')

mom1d = daily_us_eod_price_feat('mom','1d')
mom1w = daily_us_eod_price_feat('mom','1w')
mom1m = daily_us_eod_price_feat('mom','1m')
mom6m = daily_us_eod_price_feat('mom','6m')
mom12m = daily_us_eod_price_feat('mom','12m')

mom1m['chmom1m'] = mom1m.groupby('ticker')['mom1m'].diff(periods=1)

mom6m['chmom6m'] = mom6m.groupby('ticker')['mom6m'].diff(periods=1)
mom12m['chmom12m'] = mom12m.groupby('ticker')['mom12m'].diff(periods=1)


retvol1m = daily_us_eod_price_feat('retvol','1m')
retvol12m = daily_us_eod_price_feat('retvol','12m')
maxret1m = daily_us_eod_price_feat('maxret','1m')
maxret12m = daily_us_eod_price_feat('maxret','12m')

ema1m = daily_us_eod_price_feat('ema','1m')


RSI = us_eod_Adj_Close.apply(featGen.RSI, axis=0).fillna(method='ffill').unstack().reset_index(name='RSI')
MACD1m12m = us_eod_Adj_Close.apply(featGen.MACD, axis=0).fillna(method='ffill').unstack().reset_index(name='MACD1m12m')

return1m = us_eod_Adj_Close.pct_change(20).unstack().reset_index(name='return1m')
emaret1m = us_eod_Adj_Close.pct_change(20).rolling(20).mean().unstack().reset_index(name='emaret1m')
fwd_return1m = us_eod_Adj_Close.pct_change(20).shift(-20).unstack().reset_index(name='fwd_return1m')


def mrm_c(std, vol):
    value = np.tanh((10 / vol) * 50 * std)
    value[value < -0.8] = -1
    value[value > 0.8] = 1
    value[(value >= -0.8) & (value <= 0.8)] = 0
    return value

def get_norm_side(mean, vol, ret, z ):
    side = winsorize(ret, limits = [0.025,0.025])
    side[(ret - mean)/ np.sqrt(vol) > z] = 1
    side[(ret - mean)/ np.sqrt(vol) < -z] = -1
    side[((ret - mean)/ np.sqrt(vol) >= -z) & ((ret - mean)/ np.sqrt(vol) <= z)] = 0
    return side

unstack_adj_close = us_eod_Adj_Close.unstack().reset_index(name='adj_close')

# side = fwd_return1m.apply(lambda x: get_norm_side(x.ema1m, x.retvol1m, x.fwd_return, 1.645), axis=1)
dfs = [unstack_adj_close, mom1d, mom1w, mom1m, mom6m, mom12m, retvol1m, retvol12m, maxret1m,
       maxret12m, ema1m, RSI, MACD1m12m, return1m, emaret1m, fwd_return1m]


name_dfs = ['ticker', 'Date', 'adj_close', 'mom1d', 'mom1w', 'mom1m', 'chmom1m', 'mom6m', 'chmom6m', 'mom12m', 'chmom12m', 'retvol1m', 'retvol12m', 'maxret1m',
       'maxret12m', 'ema1m', 'RSI', 'MACD1m12m', 'return1m', 'emaret1m', 'fwd_return1m']

unstack_df = reduce(lambda X, x: pd.merge(X, x,  how='left', left_on=['ticker','Date'], right_on = ['ticker','Date']) ,dfs)

unstack_df.columns = name_dfs
unstack_df.index = pd.to_datetime(unstack_df.Date)

unstack_df['side'] = get_norm_side(unstack_df.emaret1m, unstack_df.retvol1m, unstack_df.fwd_return1m, 1.645)

print(unstack_df[['ticker','ema1m', 'retvol1m', 'fwd_return1m','side']])
unstack_df.to_csv('data/unstack_us_eod_with_feat.csv')

'''