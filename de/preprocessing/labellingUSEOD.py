from de.preprocessing.preUSEOD import preUSEOD
from tools.labelling_Marcos import getEvents, getDailyVol
'''
I forgot what this do

'''

pre = preUSEOD('data/equities/US_EOD_20191019.pkl', 'data/equities/preprocessed/us_eod_adj_close.pkl', 'data/equities/revolut_tickers.csv')
df_pre_useod = pre.pivot('Adj_Close').fillna(method='ffill')



