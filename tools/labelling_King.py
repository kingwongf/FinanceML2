import pandas as pd
import numpy as np

df = pd.Series([100,101,102,103,104,105,120,130,99,99,99,99,150])
def predict_ret(close, lookahead_period, min_ret=0):
    ret = np.log(close).diff(lookahead_period).dropna()
    # bin = np.sign(ret)

    bin = ret.where(ret.abs() < min_ret, 0, np.sign(ret))
    out = pd.DataFrame(index=close.index)
    out['bin'] = bin.shift(-lookahead_period)
    out['ret'] = ret.shift(-lookahead_period)
    return out




#def predict_side(close, lookahead_period, )

#print(predict_ret(df, 2, 0.01))