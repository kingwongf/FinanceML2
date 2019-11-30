from tools.featGen import mrm_c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x = np.arange(-1,1,0.0001)
y = mrm_c(x)
fig, axs = plt.subplots()
ax0 = axs.twinx()
ax1 = axs.twinx()
ax0.plot(x,y, linewidth=0.3)

df1 = pd.read_pickle("pre_data/feat_useod_daily_1mfwd.pkl")
actual_x = df1['fwdret'].values
actual_mrmc = mrm_c(actual_x)
ax0.scatter(actual_x, actual_mrmc,s=0.5)
ax1.hist(actual_x, log=True, bins=1000)
plt.xlabel('ret')
plt.ylabel('mrm_c')
plt.tight_layout()
plt.savefig("resources/ret_dist.png", dpi=500)

print(df1['fwdret'].describe())


