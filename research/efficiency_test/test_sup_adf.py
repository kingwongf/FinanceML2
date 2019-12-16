import numpy as np
import pandas as pd
from functools import reduce
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def max_mean(df):
    dfs = []
    for i in range(len(df.index)):
        dfs.append(df.expanding(i).apply(np.mean).fillna(0))
    return reduce(lambda X,x: X.clip(lower=x, axis=0), dfs)

def func_mean(x):
    return np.sum(x)/len(x)
def max_mean_non_np(df):
    dfs = []
    for i in range(len(df.index)):
        dfs.append(df.expanding(i).apply(func_mean).fillna(0))
    return reduce(lambda X,x: X.clip(lower=x, axis=0), dfs)

# print(max_mean_non_np(df))

def stationary(x, bool=False):
    # print(x)
    # print(len(x))
    if len(x) < 4:
        return 0 #a,_,_,_,e,_ = adfuller(x)
    else:
        a,_,_,_,e,_ = adfuller(x)
        if bool: return abs(a) > abs(e['5%'])
        else: return abs(a) - abs(e['5%']) # (abs(a) - abs(e['5%']))/abs(e['5%'])

# adfuller(np.array([1,2,3,4]))


def sup_adf(ts, bool=False):
    '''
        supremum ADF test on time series

    :param ts: time series
    :param bool: returns ADF t values if False, stationarity if True
    :return: boolean of stationary based on supremum ADF test

    '''
    dfs = []
    # print(len(df.index))
    for i in range(len(ts.index) + 1):
        # print(df.expanding(i))
        # print(i))
        dfs.append(ts.expanding(i).apply(stationary, args=(bool,)).fillna(0))
    return reduce(lambda X,x: X.clip(lower=x, axis=0), dfs)

ret = pd.Series(np.random.rand(100))
price = ret.cumprod().to_frame('price')*100
price.loc[50, 'price'] = 500
price.loc[80, 'price'] = 700

price['adf'] = sup_adf(price.price, True)
# df = pd.Series([1,2,1,2,1,2])

price.to_csv("research/efficiency_test/test_adf.csv")
# print(sup_adf(price))
ax1 = price.price.plot(c='b')
ax2 = ax1.twinx()
price.adf.plot(c='r',ax=ax2)
plt.legend()
plt.show()

# print(sup_adf(df, True))

# dfs = max_mean(df)


# print(df)
# print(dfs, sep='\n')
#print(dfs)