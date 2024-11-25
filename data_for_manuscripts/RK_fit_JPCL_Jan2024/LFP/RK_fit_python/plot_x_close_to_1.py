import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

R = 8.314
T = 300.0
F = 96485.0
global n_RK # order of RK
n_RK = 30

def U_RK(x,*L):
    """
    L = [Omega0, Omega1, ... Omegan, U0]
    """
    if len(L) == 1:
        L = L[0]
    summation = L[-1] # U0
    for i in range(0,n_RK):
        summation += L[i]*((1.-2.*x)**(i+1)-2.*i*x*(1.-x)*(1.-2.*x)**(i-1))
    s = R*T/F*np.log(x/(1.-x))
    return summation + s # TODO BUG sign error????


# data from NMat Fig 2a, LFP
df_discharge = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
df_discharge = df_discharge.sort_values(0)
discharge_data = df_discharge.to_numpy()

df_charge = pd.read_csv("Charge_NMat_Fig2a_even_distribution.csv",header=None)
df_charge = df_charge.sort_values(0)
charge_data = df_charge.to_numpy()


"""
Fit the discharging curve first, and try to change 
least RK params as possible to fit the charging curve
"""

# discharge
x_discharge = discharge_data[:,0]/169.91 # divided by the theoretical capacity of LFP
U_discharge = discharge_data[:,1]
# x_discharge = x_discharge[4:]
# U_discharge = U_discharge[4:]
Lopt_discharge, _ = curve_fit(U_RK,x_discharge,U_discharge,p0=np.ones(n_RK+1))
# print(Lopt_discharge, len(Lopt_discharge))
Lopt_discharge = np.array(Lopt_discharge)
x_close_to_1 = np.array([0.938, 0.939, 0.940, 0.941, 0.942, 0.943, 0.944, 0.945, 0.95, 0.96, 0.97])
x_all = np.concatenate((x_discharge, x_close_to_1))
U_discharge_RK_all = U_RK(x_all, Lopt_discharge)
np.savez("RK_with_close_to_1.npz", x=x_all, y=U_discharge_RK_all) # can be load as data=np.load("RK.npz"), SOC = data['x'], OCV_pred_RK = data['y']


# figure
# plt.figure(figsize=(5.5,4))
# discharge
plt.plot(x_discharge,U_discharge,'bo',label="Discharge Voltage (Experimental)", markersize=1.5)
plt.plot(x_all,U_discharge_RK_all,'b-',label="Discharge Voltage (RK)")
# charge
# plt.plot(x_charge,U_charge,'ro',label="Charge Voltage (Experimental)", markersize=1.5)
# plt.plot(x_charge,U_charge_RK,'r-',label="Charge RK")
plt.legend(fontsize=8)
plt.xlim([0.0,1.0])
plt.ylim([2.7,4.1])
plt.xlabel("x", fontsize=12, fontname='Arial')
plt.xticks(fontsize=12, fontname='Arial')
plt.ylabel("U(x)", fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')
plt.show()
# plt.savefig('rk_fit.png') # , bbox_inches='tight'
