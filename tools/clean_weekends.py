import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as matplotticker

def get_Nind(df):
    date = df.index.astype('O')
    N = len(df)
    ind = np.arange(N)
    return N, ind, date

def format_date(df, x, pos=None):
    N = len(df)
    thisind = np.clip(int(x + 0.5), 0, N - 1)
    return date[thisind].strftime('%Y-%m-%d')

