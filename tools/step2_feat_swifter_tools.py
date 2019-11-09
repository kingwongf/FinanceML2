import pandas as pd
from tools import featGen
import pandas as pd

from tools import featGen

print(pd.__version__)

# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199



## TODO momentum and change of momentun


def feat_ticker(close_df, closes, ticker, ticker_close, pred_freq):

    ''' close_df refers to one ticker/ pair dataframe with only one column, close_df[ticker_close]
        closes refers to all pairs df
    '''

    ## TODO mom and chmom

    '''
    D = 24*60 => 1440
    H = 60
    '''
    D = 1440
    H = 60
    min = 1

    mom = close_df.swifter.apply(featGen.momentum, axis=0, args=(D,)).fillna(method='ffill')

    # print(close_df.index)

    # mom.to_csv('testing_mom.csv')
    mom.columns = ['mom1d']
    mom['close'] = close_df
    mom['mom5d'] = close_df.apply(featGen.momentum, axis=0, args=(5*D, )).fillna(method='ffill')

    # mom.to_csv('testing_mom.csv')

    mom['mom10d'] = close_df.swifter.apply(featGen.momentum, axis=0, args=(10*D, )).fillna(method='ffill')

    mom['mom5h'] = close_df.swifter.apply(featGen.momentum, axis=0, args=(5*H, )).fillna(method='ffill')
    mom['mom1h'] = close_df.swifter.apply(featGen.momentum, axis=0, args=(H, )).fillna(method='ffill')
    mom['mom10h'] = close_df.swifter.apply(featGen.momentum, axis=0, args=(10*H, )).fillna(method='ffill')

    mom['mom30min'] = close_df.swifter.apply(featGen.momentum, axis=0, args=(30,)).fillna(method='ffill')
    mom['mom15min'] = close_df.swifter.apply(featGen.momentum, axis=0, args=(15, )).fillna(method='ffill')

    mom['chmom1d'] = mom.mom1d.diff(periods=1)
    mom['chmom5d'] = mom.mom5d.diff(periods=1)
    mom['chmom10d'] = mom.mom10d.diff(periods=1)

    mom['chmom5h'] = mom.mom5h.diff(periods=1)
    mom['chmom1h'] = mom.mom1h.diff(periods=1)
    mom['chmom10h'] = mom.mom10h.diff(periods=1)

    mom['chmom30min'] = mom.mom30min.diff(periods=1)
    mom['chmom15min'] = mom.mom15min.diff(periods=1)

    ## TODO ind mom
    # ind_currencies = set(chain.from_iterable([[x[:3], x[3:]] for x in tickers]))
    ind_currencies_top, ind_currencies_bottom = closes[[col for col in closes.columns.tolist() if col.startswith(ticker[:3])]], \
                                                closes[[col for col in closes.columns.tolist() if
                                                        ticker[:3] in col[:3]]]

    # print(ind_currencies_top.columns, ind_currencies_bottom.columns)

    # print('ind_currencies_top', ind_currencies_top.index)

    ind_mom = ind_currencies_top.apply(featGen.momentum, axis=0, args=(15, )).fillna(method='ffill').mean(axis=1, skipna=True).to_frame()

    ind_mom.columns =['top_ind_mom15min']

    ind_mom['top_ind_mom30min'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(30, )).fillna(method='ffill').mean(axis=1, skipna=True).rename('top_ind_mom30min')

    # print(ind_mom.index)
    # ind_mom.to_csv('testing_indemom.csv')



    ind_mom['top_ind_mom1h'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(1*H, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom5h'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(5*H, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom10h'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(10*H, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom1d'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(1*D, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom5d'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(5*D, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom10d'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(10*D, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom15min'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(15, ))\
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom30min'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(30, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom1h'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(1*H, ))\
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom5h'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(5*H, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom10h'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(10*H, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom1d'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(1*D, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom5d'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(5*D, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom10d'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(10*D, )) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    # print('mom', mom.index)
    # print('ind mom', ind_mom.index)

    feat_df = pd.merge_asof(mom.sort_index(), ind_mom.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms'))

    ## TODO return vol

    feat_df['retvol1d'] = close_df.swifter.apply(featGen.retvol, axis=0, args=(D,)).fillna(
        method='ffill')
    feat_df['retvol5d'] = close_df.swifter.apply(featGen.retvol, axis=0, args=(5*D,)).fillna(
        method='ffill')
    feat_df['retvol10d'] = close_df.swifter.apply(featGen.retvol, axis=0, args=(10*D,)).fillna(
        method='ffill')

    feat_df['retvol30min'] = close_df.swifter.apply(featGen.retvol, axis=0, args=(30,)).fillna(method='ffill')
    feat_df['retvol15min'] = close_df.swifter.apply(featGen.retvol, axis=0, args=(15,)).fillna(method='ffill')

    ## TODO maxret

    feat_df['maxret1d'] = close_df.swifter.apply(featGen.maxret, axis=0, args=(D,)).fillna(method='ffill')
    feat_df['maxret5d'] = close_df.swifter.apply(featGen.maxret, axis=0, args=(5*D,)).fillna(method='ffill')

    ## TODO datetime feat

    feat_df['dayofweek'] = pd.to_numeric(feat_df.index.dayofweek)

    ## TODO add ticker

    feat_df['ticker'] = ticker

    ## TODO add TA

    feat_df['RSI'] = close_df.swifter.apply(featGen.RSI, axis=0).fillna(method='ffill')
    feat_df['stochRSI'] = close_df.swifter.apply(featGen.stochRSI, axis=0).fillna(method='ffill')
    feat_df['EMA'] = close_df.swifter.apply(featGen.ema, axis=0, args=(None,0.8,)).fillna(method='ffill')

    ## TODO add label/ target, maybe change func to parse in the future

    # feat_df['target'] = close_df.swifter.apply(featGen.side, n=pred_freq)
    feat_df['target'] = close_df.swifter.apply(featGen.ret, n=pred_freq)

    feat_df['prev_ret'] = feat_df['target'].shift(1)


    # print('feat index', feat_df.index)

    return feat_df

