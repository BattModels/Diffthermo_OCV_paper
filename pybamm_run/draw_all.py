import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl  
mpl.rc('font',family='Arial')

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

c_rates = [0.05, 0.5, 1.0, 2.0] 
c_rates_label = ["C/20", "C/2", "1C", "2C"]
filenames = ["LFP_25degC_Co20.csv", "LFP_25degC_Co2.csv", "LFP_25degC_1C.csv", "LFP_25degC_2C.csv"]
ax_i_j=[[0,0],[0,1],[1,0],[1,1]]
# colors_diffthermo = ['r-', 'g-', 'b-']
# colors_Prada = ['r-.', 'g-.', 'b-.']

rmses = []

for i in range(0, len(c_rates)):
    # load true data
    filename = filenames[i]
    data = pd.read_csv(
        "data_from_AboutEnergy/validation/" + filename,
        comment="#",
    ).to_numpy()
    voltage_data = data[:, [0, 2]] # split out time [s] vs voltage [V]
    # split out time [s] vs current [A]
    current_data = data[:, [0, 1]]
    c_rate = c_rates[i]
    # this work (diffthermo)
    npz_name = "custom_LFP_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t = data['t']
    V = data['V']
    rmse = data['rmse']
    rmses.append(rmse)
    label = c_rates_label[i] # +", This Work"
    temperature = 25
    indices = ax_i_j[i]
    ax_now = ax[indices[0]][indices[1]]
    ax_now.plot(voltage_data[:, 0], voltage_data[:, 1], "--", label=f"Experiment ({temperature}\N{DEGREE SIGN}), {c_rates_label[i]}")
    ax_now.plot(t, V, "-", label=f"Model ({temperature}\N{DEGREE SIGN}), {c_rates_label[i]}")
    ax_now.set_xlabel("Time [s]", fontsize=14)
    ax_now.set_ylabel("Voltage [V]", fontsize=14)
    ax_now.set_ylim([1.9, 3.8])
    ax_now.set_xlim([-t.max()/100, t.max()*1.01])
    ax_now.legend(fontsize=12)
    ax_now.tick_params(axis='both', which='major', labelsize=14)  
plt.savefig('Figure3.png', dpi=200, bbox_inches='tight') 
plt.close()


# plot the RNSE VS C_rate
# # rmse with customized OCV: [50.857, 125.600, 149.248, 107.122]
# # rmse with original OCV from About Energy: [6.511, 101.913, 132.986, 94.942]
import matplotlib as mpl  
mpl.rc('font',family='Arial')
plt.figure(figsize=(5.5,4))
plt.plot(c_rates,rmses, "b-*")
plt.xlabel("C Rate", fontsize=18) # fontsize = 14
plt.ylabel("RMSE [mV]", fontsize=18) # fontsize = 14
plt.xticks(fontsize=18) # fontsize = 14
plt.yticks(fontsize=18) # fontsize = 14
plt.xlim([0.0,2.26])
plt.savefig('Figure3_rmse.png', dpi=200, bbox_inches='tight') 
plt.close()


