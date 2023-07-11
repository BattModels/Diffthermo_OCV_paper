# convert .csv to .npz
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

fname = "Yu_Fig2a_Energies_2021"
df = pd.read_csv(fname+".csv",header=None)
df = df.sort_values(0)
data = df.to_numpy()
SOC = data[:,0]
OCV = data[:,1]
os.chdir("../")
np.savez(fname+".npz", x=SOC, y=OCV) # load: data = np.load("test.npz"), SOC = data['x'], OCV = data['y']

# plot figure 
plt.figure(figsize=(5,4))
plt.plot(SOC, OCV, 'r-')
plt.xlim([0,1])
# plt.ylim([1.5, 5.0])
# plt.legend()
plt.xlabel("SOC")
plt.ylabel("OCV")
fig_name = "Pred.png" 
plt.show()
# plt.savefig(fig_name, bbox_inches='tight')
# plt.close()
