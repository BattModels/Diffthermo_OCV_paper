import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl  
mpl.rc('font',family='Arial')

plt.figure(figsize=(5.5,4))

c_rates = [0.5, 1.0, 2.0] 
c_rates_label = ["C/2", "1C", "2C"]
colors_diffthermo = ['r-', 'g-', 'b-']
colors_Prada = ['r-.', 'g-.', 'b-.']

for i in range(0, len(c_rates)):
    c_rate = c_rates[i]
    # this work (diffthermo)
    npz_name = "custom_LFP_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    SoC = data['SoC']
    V = data['V']
    label = c_rates_label[i] # +", This Work"
    plt.plot(SoC,V,colors_diffthermo[i],label=label)
    # # Prada2013
    # npz_name = "Prada2013_LFP_c_rate_%.4f.npz" %(c_rate)
    # data=np.load(npz_name)
    # SoC = data['SoC']
    # V = data['V']
    # label = c_rates_label[i]+", Prada2013"
    # plt.plot(SoC,V,colors_Prada[i],label=label)
plt.xlabel("SOC", fontsize=20) # fontsize = 14
plt.ylabel("Terminal Voltage (V)", fontsize=18) # fontsize = 14
plt.xticks(fontsize=20) # fontsize = 14
plt.yticks(fontsize=20) # fontsize = 14
plt.xlim([0.0,1.0])
plt.legend(fontsize=16) # fontsize = 10
plt.savefig('Figure3.png', dpi=200, bbox_inches='tight') 
plt.close()

