import numpy as np
from scipy.stats.mstats import zscore, winsorize
import pandas as pd

#print(pd.__version__)
#price = pd.read_csv("1min_price_EURTRY_2019-07-20_1min.csv")

#price.index = pd.to_datetime(price['date'])

#df = df.sort_index()
#name = 'EMA_' % span

def ema(close, span=None, alpha=None):
    ema = close.ewm(span=span, alpha=alpha,adjust=False,ignore_na=False).mean()
    return ema

def ama(close, span=None, alpha=None):
    ama = close.ma(span=span, alpha=alpha,adjust=False,ignore_na=False).mean()
    return ama

def relEMA(fast_ema, slow_ema):
    return fast_ema/ slow_ema

def RSI(close, period=14):
    delta = close.diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    rollUp = dUp.ewm(span=period).mean()
    rollDown = dDown.abs().ewm(span=period).mean()
    rsi = rollUp/ rollDown
    RSI = 100.0 - (100.0 / (1.0 + rsi))
#    print(RSI.name)
    return RSI

def stochRSI(close, period=14):
    rsi = RSI(close, period)
    rsiLow = rsi.rolling(period).min()
    rsiHigh = rsi.rolling(period).max()
    K = 100*(rsi - rsiLow)/ (rsiHigh - rsiLow)
    D = K.rolling(3).mean()
    return K, D

def stochRSI_K(close, period=14):
    rsi = RSI(close, period)
    rsiLow = rsi.rolling(period).min()
    rsiHigh = rsi.rolling(period).max()
    K = 100*(rsi - rsiLow)/ (rsiHigh - rsiLow)
    D = K.rolling(3).mean()
    return K

def stochRSI_D(close, period=14):
    rsi = RSI(close, period)
    rsiLow = rsi.rolling(period).min()
    rsiHigh = rsi.rolling(period).max()
    K = 100*(rsi - rsiLow)/ (rsiHigh - rsiLow)
    D = K.rolling(3).mean()
    return D

def MACD(close, n=(20,250)):
    short_n, long_n = n
    shortEma = close.ewm(span=short_n, min_periods=n[1]).mean()
    longEma = close.ewm(span=long_n, min_periods=n[1]).mean()
    macd = shortEma - longEma
    return macd

def momentum(close, n=1):
    # def momentum(close, n=1, freq='D'):
#     shifted_idx = close.index.shift(n, freq='D')
#     print(close.shift(n, freq=freq))
    # print(len(close))
    # print("close", close.index)
    prev_close = close.shift(n, axis=0)
    # print(prev_close)
    mom = close/prev_close -1
    # print("mom", mom.index)
    # print(mom)
    return mom

def chmom(close,n=1):
    return momentum(close, n).diff(1)

def retvol(close, period=1):
    # def retvol(close, period='1d'):
    ret = np.log(close).diff()
    vol = ret.rolling(period, min_periods=1).std()**2
    return vol

def maxret(close, period=1):
    # def maxret(close, period='1d'):
    ret = np.log(close).diff()
    max_ret = ret.rolling(period, min_periods=1).max()
    return max_ret

def ret(close, n=1):
    ret_ = np.log(close).diff(n)
    return ret_

def emaret(close, n=1):
    return ret(close, n=1).rolling(n).mean()

def fwdret(close, n=1):
    return ret(close, n=1).shift(-n)

def side(close, n=1):
    return np.sign(ret(close, n=n))

def ema_side(ret, ema_ret):
    return np.sign(ret - ema_ret)


def get_norm_side(ret, stats=(0,0,1.645)):
    mean, vol, z = stats
    side = winsorize(ret, limits=[0.025, 0.025])
    side[(ret - mean) / np.sqrt(vol) > z] = 1
    side[(ret - mean) / np.sqrt(vol) < -z] = -1
    side[((ret - mean) / np.sqrt(vol) >= -z) & ((ret - mean) / np.sqrt(vol) <= z)] = 0
    return side



def util_winsor5(x):
    return (winsorize(x,limits=[0.05, 0.05]))

def nanzscore(a):
    if len(a[~np.isnan(a)]) > 10:
        z = a                    # initialise array for zscores
        z[~np.isnan(a)] = util_winsor5(z[~np.isnan(z)])
        z[~np.isnan(a)] = zscore(z[~np.isnan(z)])
    else:
        z = a
        z[~np.isnan(a)] = np.repeat(0, len(a[~np.isnan(a)]), axis=0)
    return(z)

def nanzscore_gp(a,gp):
    pdf  = pd.DataFrame({'a':[a], 'gp':[gp]})
    pdfz = pdf.groupby('gp').a.transform(nanzscore)
    return((pdfz))

def mrm_c(std):
    return(  (0.95-np.tanh(std/1.3)**2) * (std) )


def moskowitz_func(x):
  # Moskowitz et al. 2012
  return x * np.exp( - (x**2)/4) /0.89

def tanh_func(x):
  return np.tanh(x)


def cat_mrm_c(std, vol):
    value = np.tanh((10 / vol) * 50 * std)
    value[value < -0.8] = -1
    value[value > 0.8] = 1
    value[(value >= -0.8) & (value <= 0.8)] = 0
    return value

def step_func(x, threshold):
    x[x < -threshold] = -1
    x[x > threshold] = 1
    x[(x > -threshold) &((x < threshold))] = 0
    return x
def log_ts(x,n=None):
    return np.log(x)
#K, D = stochRSI(price['4. close'])

#print(MACD(price['4. close']))

#print(price[['EMA_5', 'pandas_5days_EMA']])

#print(stochRSI()