import numpy as np


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
    RSI = RSI.rename('RSI ' + close.name, inplace=True)
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

def MACD(close):
    shortEma = close.ewm(adjust=True, alpha=0.15).mean()
    longEma = close.ewm(adjust=True, alpha=0.075).mean()
    macd = shortEma - longEma
    return macd

def momentum(close, n=1):
    # def momentum(close, n=1, freq='D'):
#     shifted_idx = close.index.shift(n, freq='D')
#     print(close.shift(n, freq=freq))
    # print(len(close))
    # print("close", close.index)
    prev_close = close.shift(n, axis=0)
    # print(prev_close.columns)
    mom = close/prev_close -1
    # print("mom", mom.index)
    # print(mom)
    return mom

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
    ret_ = np.log(close).diff(-n).shift(n)
    return ret_
def side(close, n=1):
    ret_ = ret(close, n=n)
    side_ = np.sign(ret_)
    return side_
#K, D = stochRSI(price['4. close'])

#print(MACD(price['4. close']))

#print(price[['EMA_5', 'pandas_5days_EMA']])

#print(stochRSI()