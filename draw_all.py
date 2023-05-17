"""
Draw all results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

working_dir = os.getcwd()

import matplotlib as mpl  
from matplotlib.ticker import FormatStrFormatter
mpl.rc('font',family='Arial')

# define canvas
fig, ax = plt.subplots(4, 3, figsize=(22.5, 30))

# LFP, true OCV
os.chdir("LFP/Discharge_4_RK_params")
df_discharge = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
df_discharge = df_discharge.sort_values(0)
data = df_discharge.to_numpy()
SOC_true_LFP = data[:,0]/169.91 # divided by the theoretical capacity of LFP
OCV_true_LFP = data[:,1]
# LFP, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LFP = data['x']
OCV_diffthermo_LFP = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[0][0]
ax_now.plot(SOC_true_LFP, OCV_true_LFP, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LFP, OCV_diffthermo_LFP, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.45, 4.2])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('LFP, 4 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# graphite, true OCV
os.chdir("graphite/7_RK_params")
df = pd.read_csv("graphite.csv",header=None)
data = df.to_numpy()
SOC_true_graphite = data[:,0]
OCV_true_graphite = data[:,1]
# graphite, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_graphite = data['x']
OCV_diffthermo_graphite = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[0][1]
ax_now.plot(SOC_true_graphite, OCV_true_graphite, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_graphite, OCV_diffthermo_graphite, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([0.0, 0.65])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('Graphite, 7 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# LixFeSiO4, true OCV
os.chdir("LixFeSiO4/6_RK_params")
df = pd.read_csv("Discharge_NMat_Fig1a.csv",header=None)
data = df.to_numpy()
SOC_true_LixFeSiO4 = data[:,0]-1.0 # Lix, x \in [1,2]
OCV_true_LixFeSiO4 = data[:,1]
# LixFeSO4, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LixFeSiO4 = data['x']
OCV_diffthermo_LixFeSiO4 = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[0][2]
ax_now.plot(SOC_true_LixFeSiO4, OCV_true_LixFeSiO4, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LixFeSiO4, OCV_diffthermo_LixFeSiO4, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.17, 3.85])
ax_now.set_xlim([0.2, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('Li$_{x}$FeSiO$_{4}$, 6 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# LCO, true OCV
os.chdir("LCO/8_RK_params")
df = pd.read_csv("LCO_Carlier2012JES_Fig4a.csv",header=None)
data = df.to_numpy()
SOC_true_LCO = data[:,0]
OCV_true_LCO = data[:,1]
# LCO, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LCO = data['x']
OCV_diffthermo_LCO = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[1][0]
ax_now.plot(SOC_true_LCO, OCV_true_LCO, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LCO, OCV_diffthermo_LCO, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.9, 4.8])
ax_now.set_xlim([0.15, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('LCO, 8 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# LMP, true OCV
os.chdir("LMP/5_RK_params")
df = pd.read_csv("LiMnPO4.csv",header=None)
data = df.to_numpy()
SOC_true_LMP = data[:,0]/170.87 # theoretical capacity of LiMnPO4
OCV_true_LMP = data[:,1]
# LMP, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LMP = data['x']
OCV_diffthermo_LMP = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[1][1]
ax_now.plot(SOC_true_LMP, OCV_true_LMP, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LMP, OCV_diffthermo_LMP, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.2, 4.7])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('LMP, 5 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# LMFP, true OCV
os.chdir("LMFP/7_RK_params")
df = pd.read_csv("LiFeMnPO4.csv",header=None)
data = df.to_numpy()
SOC_true_LMFP = data[:,0]/170.48 # theoretical capacity of LiMn0.5Fe0.5PO4
OCV_true_LMFP = data[:,1]
# LMFP, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LMFP = data['x']
OCV_diffthermo_LMFP = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[1][2]
ax_now.plot(SOC_true_LMFP, OCV_true_LMFP, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LMFP, OCV_diffthermo_LMFP, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.0, 4.2])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('LiMn$_{0.5}$Fe$_{0.5}$PO$_{4}$, 7 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)


# LTO, true OCV
os.chdir("LTO/4_RK_params")
df = pd.read_csv("LTO.csv",header=None)
data = df.to_numpy()
SOC_true_LTO = 1.0-data[:,0]/175 # theoretical capacity of LTO
OCV_true_LTO = data[:,1]
# LTO, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LTO = data['x']
OCV_diffthermo_LTO = data['y']
# print(SOC_true_LTO, OCV_true_LTO) # AMYAO DEBUG
# print("******")
# print(SOC_diffthermo_LTO, OCV_diffthermo_LTO) # AMYAO DEBUG
# exit()
os.chdir(working_dir)
# draw
ax_now = ax[2][0]
ax_now.plot(SOC_true_LTO, OCV_true_LTO, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LTO, OCV_diffthermo_LTO, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([0.7, 2.6])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('Li$_{4/3}$Ti$_{5/3}$O$_{4}$, 4 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)


# Si, true OCV
os.chdir("Si/6_RK_params")
df = pd.read_csv("Si.csv",header=None)
data = df.to_numpy()
SOC_true_Si = data[:,0]/3600 # theoretical capacity of Si
OCV_true_Si = data[:,1]
# Si, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_Si = data['x']
OCV_diffthermo_Si = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[2][1]
ax_now.plot(SOC_true_Si, OCV_true_Si, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_Si, OCV_diffthermo_Si, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([0.0, 0.8])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('Si, 6 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)


# LMO, true OCV
os.chdir("LMO/5_RK_params")
df = pd.read_csv("LMO.csv",header=None)
data = df.to_numpy()
SOC_true_LMO = data[:,0]
OCV_true_LMO = data[:,1]
# LMO, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_LMO = data['x']
OCV_diffthermo_LMO = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[2][2]
ax_now.plot(SOC_true_LMO, OCV_true_LMO, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_LMO, OCV_diffthermo_LMO, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([3.38, 4.24])
ax_now.set_xlim([0.15 ,1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('LMO, 5 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)


# NCA, true OCV
os.chdir("NCA/6_RK_params")
df = pd.read_csv("nca_ocp_Kim2011_data.csv",header=None)
data = df.to_numpy()
SOC_true_NCA = data[:,0]
OCV_true_NCA = data[:,1]
# NCA, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_NCA = data['x']
OCV_diffthermo_NCA = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[3][0]
ax_now.plot(SOC_true_NCA, OCV_true_NCA, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_NCA, OCV_diffthermo_NCA, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.9, 4.4])
ax_now.set_xlim([0.3, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('NCA, 6 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# NCO, true OCV
os.chdir("NCO/6_RK_params")
df = pd.read_csv("nco_ocp_Ecker2015.csv",header=None)
data = df.to_numpy()
SOC_true_NCO = data[:,0]
OCV_true_NCO = data[:,1]
# NCO, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_NCO = data['x']
OCV_diffthermo_NCO = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[3][1]
ax_now.plot(SOC_true_NCO, OCV_true_NCO, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_NCO, OCV_diffthermo_NCO, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([3.3, 4.7])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('NCO, 6 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



# NMC, true OCV
os.chdir("NMC/6_RK_params")
df = pd.read_csv("nmc_LGM50_ocp_Chen2020.csv",header=None)
data = df.to_numpy()
SOC_true_NMC = data[:,0]
OCV_true_NMC = data[:,1]
# NMC, diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo_NMC = data['x']
OCV_diffthermo_NMC = data['y']
os.chdir(working_dir)
# draw
ax_now = ax[3][2]
ax_now.plot(SOC_true_NMC, OCV_true_NMC, "k-", label="True Value")
ax_now.plot(SOC_diffthermo_NMC, OCV_diffthermo_NMC, "b--", label="This Work")
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([3.51, 4.42])
ax_now.set_xlim([0.2, 1.0])
ax_now.legend(fontsize=16)
ax_now.set_title('NMC, 6 RK Parameters',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)



plt.savefig('Figure_raw.png', dpi=200, bbox_inches='tight') 
plt.close()

