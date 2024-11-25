import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl  
mpl.rc('font',family='Arial')


# no error plot
fig, ax = plt.subplots(2, 2, figsize=(13.5, 11))
c_rates = [0.05, 2.0] 
c_rates_label = ["C/20", "2C"]
filenames = ["LFP_25degC_Co20.csv", "LFP_25degC_2C.csv"]
ax_i_j=[[0,0],[0,1]]
rmses_diffthermo = []
rmses_AboutEnergy = []
for i in range(0, len(c_rates)):
    # load true data
    filename = filenames[i]
    data = pd.read_csv(
        "LFP/pybamm_run/data_from_AboutEnergy/validation/" + filename,
        comment="#",
    ).to_numpy()
    voltage_data = data[:, [0, 2]] # split out time [s] vs voltage [V]
    # split out time [s] vs current [A]
    current_data = data[:, [0, 1]]
    c_rate = c_rates[i]
    # this work (diffthermo)
    npz_name = "LFP/pybamm_run/custom_LFP_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t = data['t']
    V = data['V']
    rmse = data['rmse']
    rmses_diffthermo.append(rmse)
    # AboutEnergy
    npz_name = "LFP/pybamm_run/AboutEnergy_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t_a = data['t']
    V_a = data['V']
    rmse_a = data['rmse']
    rmses_AboutEnergy.append(rmse_a)
    label = c_rates_label[i] # +", This Work"
    temperature = 25
    indices = ax_i_j[i]
    ax_now = ax[indices[0]][indices[1]]
    ax_now.plot(voltage_data[:, 0], voltage_data[:, 1], "k-", label=f"Experiment, {c_rates_label[i]}", linewidth=3)
    ax_now.plot(t, V, "b--", label=f"PyBamm Simulation, {c_rates_label[i]}", linewidth=3)
    # ax_now.plot(t_a, V_a, "r--", label=f"AboutEnergy, {c_rates_label[i]}")
    ax_now.set_xlabel("Time (s)", fontsize=20)
    ax_now.set_ylabel("Voltage (V)", fontsize=20)
    ax_now.set_ylim([1.9, 3.8])
    ax_now.set_xlim([-t.max()/100, t.max()*1.01])
    ax_now.legend(fontsize=16, frameon=False)
    ax_now.tick_params(axis='both', which='major', labelsize=20)  
    for axis in ['top','bottom','left','right']:
        ax_now.spines[axis].set_linewidth(3)
    ax_now.tick_params(width=3)

c_rates = [0.05, 2.0] 
c_rates_label = ["C/20","2C"]
filenames = ["NMC_25degC_Co20.csv", "NMC_25degC_2C.csv"]
ax_i_j=[[1,0],[1,1]]
rmses_diffthermo = []
rmses_AboutEnergy = []
for i in range(0, len(c_rates)):
    # load true data
    filename = filenames[i]
    data = pd.read_csv(
        "graphite/pybamm_run/data_from_AboutEnergy/validation/" + filename,
        comment="#",
    ).to_numpy()
    voltage_data = data[:, [0, 2]] # split out time [s] vs voltage [V]
    # split out time [s] vs current [A]
    current_data = data[:, [0, 1]]
    c_rate = c_rates[i]
    # this work (diffthermo)
    npz_name = "graphite/pybamm_run/custom_NMC_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t = data['t']
    V = data['V']
    rmse = data['rmse']
    rmses_diffthermo.append(rmse)
    # AboutEnergy
    npz_name = "graphite/pybamm_run/AboutEnergy_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t_a = data['t']
    V_a = data['V']
    rmse_a = data['rmse']
    rmses_AboutEnergy.append(rmse_a)
    label = c_rates_label[i] # +", This Work"
    temperature = 25
    indices = ax_i_j[i]
    ax_now = ax[indices[0]][indices[1]]
    ax_now.plot(voltage_data[:, 0], voltage_data[:, 1], "k-", label=f"Experiment, {c_rates_label[i]}", linewidth=3)
    ax_now.plot(t, V, "b--", label=f"PyBamm Simulation, {c_rates_label[i]}", linewidth=3)
    # ax_now.plot(t_a, V_a, "r--", label=f"AboutEnergy, {c_rates_label[i]}")
    ax_now.set_xlabel("Time (s)", fontsize=20)
    ax_now.set_ylabel("Voltage (V)", fontsize=20)
    ax_now.set_ylim([2.5, 4.5])
    ax_now.set_xlim([-t.max()/100, t.max()*1.01])
    ax_now.legend(fontsize=16, frameon=False)
    ax_now.tick_params(axis='both', which='major', labelsize=20)  
    for axis in ['top','bottom','left','right']:
        ax_now.spines[axis].set_linewidth(3)
    ax_now.tick_params(width=3)

plt.savefig('Figure4_raw.png', dpi=200, bbox_inches='tight') 
plt.close()
