import pandas as pd
import numpy as np
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def pd_min_max_scalar(df):
	for num in df.columns.tolist():
		df[num] = (df[num] - df[num].min())/ (df[num.max].max() - df[num].min())

@timeit
def pd_apply_min_max(df):
	df[0:500] = (df[0:500] - df[0:500].min(axis=0))/ (df[0:500].max(axis=0) - df[0:500].min(axis=0))

@timeit
def np_min_max(df):
	df[0:500] = (df[0:500] - np.min(df[0:500], axis=0)) / (np.max(df[0:500], axis=0) - np.min(df[0:500], axis=0))


np_df = pd.DataFrame(np.random.rand(1000,1000))
np_df = np_min_max(np_df)

pd_df = pd.DataFrame(np.random.rand(1000,1000))
pd_df = pd_apply_min_max(pd_df)