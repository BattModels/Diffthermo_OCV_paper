import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

R = 8.314
T = 300.0
F = 96485.0
global n_RK # order of RK
n_RK = 7

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
    return summation - s # TODO BUG sign error????



df_discharge = pd.read_csv("graphite.csv",header=None)
df_discharge = df_discharge.sort_values(0)
discharge_data = df_discharge.to_numpy()

# discharge
x_discharge = discharge_data[:,0]
U_discharge = discharge_data[:,1]
# x_discharge = x_discharge[4:]
# U_discharge = U_discharge[4:]
Lopt_discharge, _ = curve_fit(U_RK,x_discharge,U_discharge,p0=np.ones(n_RK+1))
# print(Lopt_discharge, len(Lopt_discharge))
Lopt_discharge = np.array(Lopt_discharge)
U_discharge_RK = U_RK(x_discharge, Lopt_discharge)
loss_discharge = np.sqrt(np.mean((U_discharge_RK - U_discharge)**2))
print("RMSE discharge = %.4f" %(loss_discharge))
np.savez("RK.npz", x=x_discharge, y=U_discharge_RK) # can be load as data=np.load("RK.npz"), SOC = data['x'], OCV_pred_RK = data['y']


# figure
# plt.figure(figsize=(5.5,4))
# discharge
plt.plot(x_discharge,U_discharge,'bo',label="Discharge Voltage (Experimental)", markersize=1.5)
plt.plot(x_discharge,U_discharge_RK,'b-',label="Discharge Voltage (RK)")
# charge
# plt.plot(x_charge,U_charge,'ro',label="Charge Voltage (Experimental)", markersize=1.5)
# plt.plot(x_charge,U_charge_RK,'r-',label="Charge RK")
plt.legend(fontsize=8)
plt.xlim([0.0,1.0])
# plt.ylim([-0.02,0.04])
plt.xlabel("x", fontsize=12, fontname='Arial')
plt.xticks(fontsize=12, fontname='Arial')
plt.ylabel("U(x)", fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')
plt.show()
# plt.savefig('rk_fit.png') # , bbox_inches='tight'
