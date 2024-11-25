import numpy as np
import os
import matplotlib.pyplot as plt


data=np.load("RK_Plett_LTO.npz")
SOC = data['x']
OCV = data['y']

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
