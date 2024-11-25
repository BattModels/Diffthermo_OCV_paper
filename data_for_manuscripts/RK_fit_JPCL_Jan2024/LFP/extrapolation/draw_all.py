"""
Draw the extrapolation results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# regular_RK fit
os.chdir("regular_RK")
data=np.load("pred.npz")
SOC = data['x']
OCV_true = data['y']
OCV_pred_RK = data['y_pred']

os.chdir("../")

# diffthermo fit
os.chdir("thermodynamically_consistent_model")
data=np.load("pred.npz")
OCV_pred_diffthermo = data['y_pred']
os.chdir("../")

import matplotlib as mpl  
from matplotlib.ticker import FormatStrFormatter
mpl.rc('font',family='Arial')

# figure
plt.figure(figsize=(5.5,4))
ax = plt.gca()
plt.plot(SOC,OCV_true,'k-',label="True OCV", markersize=1.5)
plt.plot(SOC,OCV_pred_diffthermo,'r-.',label="This work", markersize=4)
# plt.plot(SOC_diffthermo_splines,OCV_diffthermo_splines,'c-.',label="Diffthermo w/ splines", markersize=1.5)
plt.plot(SOC,OCV_pred_RK,'g-.',label="Regular RK", markersize=1.5)
plt.legend(fontsize=10, frameon=False)
plt.xlim([0.934,0.941])
# plt.ylim([-0.02,0.04])
plt.xlabel("SOC", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("OCV (V)", fontsize=14)
plt.yticks(fontsize=14)
# plt.tight_layout()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
plt.savefig('all_fit.png', dpi=200, bbox_inches='tight') 
plt.close()
