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
    return summation + s 


# data from NMat Fig 2a, LFP
df_discharge = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
df_discharge = df_discharge.sort_values(0)
discharge_data = df_discharge.to_numpy()

"""
Fit the discharging curve first, and try to change 
least RK params as possible to fit the charging curve
"""

# discharge
"""
To prove regular RK overfits, we leave the last 10 datapoint out (train from x= 0 to 0.935, predict 0.94)
"""
x_discharge = discharge_data[0:-10,0]/169.91 # divided by the theoretical capacity of LFP
U_discharge = discharge_data[0:-10,1]
x_leaveout = discharge_data[-1,0]/169.91
U_leaveout = discharge_data[-1,1]
print(x_discharge[-1], x_leaveout)
# x_discharge = x_discharge[4:]
# U_discharge = U_discharge[4:]
Lopt_discharge, _ = curve_fit(U_RK,x_discharge,U_discharge,p0=np.ones(n_RK+1))
# print(Lopt_discharge, len(Lopt_discharge))
Lopt_discharge = np.array(Lopt_discharge)
U_discharge_RK = U_RK(x_discharge, Lopt_discharge)
loss_discharge = np.sqrt(np.mean((U_discharge_RK - U_discharge)**2))
print("RMSE discharge = %.4f" %(loss_discharge))
# np.savez("RK.npz", x=x_discharge, y=U_discharge_RK) # can be load as data=np.load("RK.npz"), SOC = data['x'], OCV_pred_RK = data['y']

"""predict on the last data point"""

U_leaveout_RK = U_RK(x_leaveout, Lopt_discharge)
print("Leaveout point: RK predicts %.4f, real value %.4f, error %.4f" %(U_leaveout_RK, U_leaveout, np.abs(U_leaveout-U_leaveout_RK)))
text = "Leaveout point: RK predicts %.4f, real value %.4f, error %.4f" %(U_leaveout_RK, U_leaveout, np.abs(U_leaveout-U_leaveout_RK))
print(text)
with open("result.txt",'w') as fout:
    fout.write(text)
exit()
