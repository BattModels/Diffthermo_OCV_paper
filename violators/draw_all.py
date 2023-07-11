"""
Draw violator
"""

import numpy as np
import matplotlib.pyplot as plt
import os

working_dir = os.getcwd()

import matplotlib as mpl  
mpl.rc('font',family='Arial')

# define canvas
fig, ax = plt.subplots(2, 3, figsize=(22.5, 15))


# Karthikeyan 2 param Margules model
data = np.load("Karthikeyan_Fig3_J_Power_Sources_2008.npz")
SOC = data['x']
OCV = data['y']
ax_now = ax[0][0]
ax_now.plot(SOC, OCV, "b-", linewidth=3) # , label="Experimental Value"
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([0.0, 0.5])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16, frameon=False)
ax_now.set_title('2 Parameter Margules',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)
for axis in ['top','bottom','left','right']:
    ax_now.spines[axis].set_linewidth(3)
ax_now.tick_params(width=3)

# Plett Skewed R-K model
data = np.load("RK_Plett_LFP.npz")
SOC = data['x']
OCV = data['y']
ax_now = ax[0][1]
ax_now.plot(SOC, OCV, "b-", linewidth=3) # , label="Experimental Value"
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.6, 3.6])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16, frameon=False)
ax_now.set_title('Skewed R-K',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)
for axis in ['top','bottom','left','right']:
    ax_now.spines[axis].set_linewidth(3)
ax_now.tick_params(width=3)

# Nejad 8th order polynomial
data = np.load("Nejad_fig1a_J_Power_Sources_2016.npz")
SOC = data['x']
OCV = data['y']
ax_now = ax[0][2]
ax_now.plot(SOC, OCV, "b-", linewidth=3) # , label="Experimental Value"
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.6, 3.6])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16, frameon=False)
ax_now.set_title('8th Order Polynomial',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)
for axis in ['top','bottom','left','right']:
    ax_now.spines[axis].set_linewidth(3)
ax_now.tick_params(width=3)


# Weng Double Exponential Function
data = np.load("Weng_Fig4_J_Power_Sources_2014.npz")
SOC = data['x']
OCV = data['y']
ax_now = ax[1][0]
ax_now.plot(SOC, OCV, "b-", linewidth=3) # , label="Experimental Value"
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
# ax_now.set_ylim([2.65, 4.3])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16, frameon=False)
ax_now.set_title('Double Exponential Function',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)
for axis in ['top','bottom','left','right']:
    ax_now.spines[axis].set_linewidth(3)
ax_now.tick_params(width=3)


# Pan 6th Order Polynomial
data = np.load("Pan_Fig8_Energy_2017.npz")
SOC = data['x']
OCV = data['y']
ax_now = ax[1][1]
ax_now.plot(SOC, OCV, "b-", linewidth=3) # , label="Experimental Value"
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([3.0, 4.5])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16, frameon=False)
ax_now.set_title('6th Order Polynomial',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)
for axis in ['top','bottom','left','right']:
    ax_now.spines[axis].set_linewidth(3)
ax_now.tick_params(width=3)

# Yu 4th Order Polynomial
data = np.load("Yu_Fig2a_Energies_2021.npz")
SOC = data['x']
OCV = data['y']
ax_now = ax[1][2]
ax_now.plot(SOC, OCV, "b-", linewidth=3) # , label="Experimental Value"
ax_now.set_xlabel("SOC", fontsize=20)
ax_now.set_ylabel("OCV (V)", fontsize=20)
ax_now.set_ylim([2.5, 4.5])
ax_now.set_xlim([0.0, 1.0])
ax_now.legend(fontsize=16, frameon=False)
ax_now.set_title('4th Order Polynomial',fontsize=20)
ax_now.tick_params(axis='both', which='major', labelsize=20)  
ax_now.locator_params(axis='both', nbins=5)
for axis in ['top','bottom','left','right']:
    ax_now.spines[axis].set_linewidth(3)
ax_now.tick_params(width=3)



plt.savefig('Figure_raw.png', dpi=200, bbox_inches='tight') 
plt.close()

