import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl  
mpl.rc('font',family='Arial')


# no error plot
fig, ax = plt.subplots(2, 2, figsize=(13.5, 11))
c_rates = [0.05, 0.5, 1.0, 2.0] 
c_rates_label = ["C/20", "C/2", "1C", "2C"]
filenames = ["LFP_25degC_Co20.csv", "LFP_25degC_Co2.csv", "LFP_25degC_1C.csv", "LFP_25degC_2C.csv"]
ax_i_j=[[0,0],[0,1],[1,0],[1,1]]
rmses_diffthermo = []
rmses_monotonic_RK=[]
rmses_AboutEnergy = []
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
    rmses_diffthermo.append(rmse)
    # monotonic_RK
    npz_name = "Monotonic_RK_LFP_c_rate_%.4f.npz" %(c_rate)
    data=np.load(npz_name)
    t_m = data['t']
    V_m= data['V']
    rmse = data['rmse']
    rmses_monotonic_RK.append(rmse)
    # AboutEnergy
    npz_name = "AboutEnergy_c_rate_%.4f.npz" %(c_rate)
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
    ax_now.plot(t, V, "b--", label=f"PyBamm This Work, {c_rates_label[i]}", linewidth=3)
    ax_now.plot(t_m, V_m, "r--", label=f"PyBamm Monotonic RK, {c_rates_label[i]}", linewidth=3)
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
plt.savefig('Figure_raw.png', dpi=200, bbox_inches='tight') 
plt.close()

# # with error plot
# fig, ax = plt.subplots(4, 2, figsize=(17, 20))
# c_rates = [0.05, 0.5, 1.0, 2.0] 
# c_rates_label = ["C/20", "C/2", "1C", "2C"]
# filenames = ["LFP_25degC_Co20.csv", "LFP_25degC_Co2.csv", "LFP_25degC_1C.csv", "LFP_25degC_2C.csv"]
# rmses_diffthermo = []
# rmses_AboutEnergy = []
# for i in range(0, len(c_rates)):   
#     # load true data
#     filename = filenames[i]
#     data = pd.read_csv(
#         "data_from_AboutEnergy/validation/" + filename,
#         comment="#",
#     ).to_numpy()
#     voltage_data = data[:, [0, 2]] # split out time [s] vs voltage [V]
#     # split out time [s] vs current [A]
#     current_data = data[:, [0, 1]]
#     c_rate = c_rates[i]
#     # this work (diffthermo)
#     npz_name = "custom_LFP_c_rate_%.4f.npz" %(c_rate)
#     data=np.load(npz_name)
#     t = data['t']
#     V = data['V']
#     rmse = data['rmse']
#     rmses_diffthermo.append(rmse)
#     # AboutEnergy
#     npz_name = "AboutEnergy_c_rate_%.4f.npz" %(c_rate)
#     data=np.load(npz_name)
#     t_a = data['t']
#     V_a = data['V']
#     rmse_a = data['rmse']
#     rmses_AboutEnergy.append(rmse_a)
#     label = c_rates_label[i] # +", This Work"
#     indices = ax_i_j[i]
#     # simulation plot
#     ax_now = ax[i][0]
#     ax_now.plot(voltage_data[:, 0], voltage_data[:, 1], "k-", label=f"Experiment, {c_rates_label[i]}")
#     ax_now.plot(t, V, "b--", label=f"PyBamm Simulation, {c_rates_label[i]}")
#     # ax_now.plot(t_a, V_a, "r--", label=f"AboutEnergy, {c_rates_label[i]}")
#     ax_now.set_xlabel("Time (s)", fontsize=20)
#     ax_now.set_ylabel("Voltage (V)", fontsize=20)
#     ax_now.set_ylim([1.9, 3.8])
#     ax_now.set_xlim([-t.max()/100, t.max()*1.01])
#     ax_now.legend(fontsize=16)
#     ax_now.tick_params(axis='both', which='major', labelsize=20)  
#     # Error plot
#     ax_now = ax[i][1]
#     # this work
#     rmse = np.sqrt( np.nanmean((voltage_data[:, 1] - V)**2) ) * 1000
#     # label = f"This Work, {c_rates_label[i]}, RMSE = "
#     label = f"{c_rates_label[i]}, RMSE = "
#     label = label + "%.3f mV" %(rmse)
#     ax_now.plot(t, (V - voltage_data[:, 1]) * 1000, "b-", label = label)
#     # # about energy
#     # rmse = np.sqrt( np.nanmean((voltage_data[:, 1] - V_a)**2) ) * 1000
#     # label = f"About Energy, {c_rates_label[i]}, RMSE = "
#     # label = label + "%.3f mV" %(rmse)
#     # ax_now.plot(t, (V_a - voltage_data[:, 1]) * 1000, "r-", label = label)
#     # settings
#     ax_now.set_xlabel("Time (s)", fontsize=20)
#     ax_now.set_ylabel("Error (mV)", fontsize=20)
#     ax_now.set_xlim([-t.max()/100, t.max()*1.01])
#     ax_now.legend(fontsize=16)
#     ax_now.tick_params(axis='both', which='major', labelsize=20)  

# plt.savefig('Figure3_with_error.png', dpi=200, bbox_inches='tight') 
# plt.close()


# plot the RNSE VS C_rate
# # rmse with diffthermo OCV: [50.857, 125.600, 149.248, 107.122]
# # rmse with monotonic RK: [103.983, 144.028, 157.955, 147.254]
import matplotlib as mpl  
mpl.rc('font',family='Arial')
plt.figure(figsize=(5.5,4))
plt.plot(c_rates,rmses_diffthermo, "b-*", label="PyBamm This Work")
plt.plot(c_rates,rmses_monotonic_RK, "r-*", label="Monotonic RK")
# plt.plot(c_rates,rmses_AboutEnergy, "k-*", label="AboutEnergy")
plt.xlabel("C Rate", fontsize=18) # fontsize = 14
plt.ylabel("RMSE (mV)", fontsize=18) # fontsize = 14
plt.xticks(fontsize=18) # fontsize = 14
plt.yticks(fontsize=18) # fontsize = 14
plt.legend(fontsize = 14)
plt.xlim([0.0,2.26])
plt.savefig('Figure3_rmse.png', dpi=200, bbox_inches='tight') 
plt.close()


