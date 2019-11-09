import numpy as np
def analytical_OLS(olsX, olsY):
    #X, y = X.dropna(), y.dropna()
    #print(X.size(), y.size())

    X_prime_y = olsX.transpose().dot(olsY)

    beta_hat = np.linalg.inv(olsX.transpose().dot(olsX)).dot(X_prime_y)


    e = pd.merge_asof(olsY, olsX.dot(beta_hat).rename("pre_e"), left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))
#    print(e)

    error = e.diff(axis=1)

    return beta_hat, error