import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from autograd import grad

import matplotlib.pyplot as plt
from scipy.optimize import minimize

def getWeights_FFD(d,thres):
    w,k=[1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_);k+=1
    return np.array(w[::-1]).reshape(-1,1)

#---------------------------------------------------------------------------

def fracDiff_FFD(series,d,thres=1e-3):
    # Constant width window (new solution)
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            test_val = series.loc[loc1,name] # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any(): continue # exclude NAs
            #print(f'd: {d}, iloc1:{iloc1} shapes: w:{w.T.shape}, series: {seriesF.loc[loc0:loc1].notnull().shape}')
            try:
                df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
            except:
                continue
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df


def get_optimal_ffd(x, ds, t=1e-5):
    cols = ['adfStat', 'pVal', 'lags', 'nObs', '95% conf']  # ,'corr']
    out = pd.DataFrame(columns=cols)

    for d in tqdm(ds):
        try:
            # dfx = fracDiff(x.to_frame(),d,thres=1e-5)
            dfx = fracDiff_FFD(x.to_frame(), d, thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx[dfx.columns[0]], maxlag=1, regression='c', autolag=None)
            out.loc[d] = list(dfx[:4]) + [dfx[4]['5%']]
        except Exception as e:
            print(f'{d} error: {e}')
            break
    return out


# ============================================

def min_get_optimal_ffd(x, d, t=1e-5):
    try:
            dfx = fracDiff_FFD(x.to_frame(), d, thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx[dfx.columns[0]], maxlag=1, regression='c', autolag=None)
            out = dfx[0] - dfx[4]['5%'] + 0.00001 ## so adfstats is optimized to be slightly more negative
    except Exception as e:
        print(f'{d} error: {e}')
    return out


# ============================================

def test_get_optimal_ffd(x, ds, t=1e-3):
    cost_funcs =[]
    for i,d in enumerate(ds):
        try:
            dfx = fracDiff_FFD(x.to_frame(), d, thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx[dfx.columns[0]], maxlag=1, regression='c', autolag=None)
            cost = dfx[0] - dfx[4]['5%']
            cost_funcs.append(cost)
            if cost_funcs[i-1] < 0 and cost_funcs[i] < cost_funcs[i-1]:
                opt_d = ds[i-1]
                return opt_d
        except Exception as e:
            print(f'{d} error: {e}')


# ============================================


'''
open_closes = pd.read_pickle("data/source_latest.pkl")

thres = 1e-5

def reporter(d):
    global ps
    print(ps)
    ps.append(d)

x0=0.05
ps=[x0]
ds = np.arange(0, 1, 0.001)
res = get_optimal_ffd(open_closes['AUDCAD 4. close'], ds)
print(res)
#res = minimize(min_get_optimal_ffd, x0, args=(open_closes['AUDCAD 4. close']), method='SLSQP',
#                tol=1e-2, callback=reporter)
#grad_d = grad(min_get_optimal_ffd(args=open_closes['AUDCAD 4. close']))


f,ax=plt.subplots()
out['adfStat'].plot(ax=ax, marker="X", markersize=10)
ax.axhline(out['95% conf'].mean(),lw=1,color='r',ls='dotted')
ax.set_title(f'min d with thresh={thres}')
ax.set_xlabel('d values')
ax.set_ylabel('adf stat');

plt.show()
min_d_index = (out['adfStat'] < out['95% conf']).idxmax()
print(min_d_index)
'''