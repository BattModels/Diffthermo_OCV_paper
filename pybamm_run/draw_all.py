import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl  
mpl.rc('font',family='Arial')

plt.figure(figsize=(5.5,4))


c_rates = [0.5, 1.0, 2.0] 

for c_rate in c_rates:
    npz_name = "custom_LFP_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t = data['t']
    V = data['V']
    plt.plot(t,V,label="C Rate = %.2f" %(c_rate))
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Terminal Voltage (V)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([0,6000])
plt.legend(fontsize=10) 
plt.savefig('Figure3.png', dpi=200, bbox_inches='tight') 
plt.close()

