import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
U(x) = U0 + RT/F*log((1-x)/x) + (1/(K*(2x-1)+1)**2) * ( \sum_{i=0}^15 Ai/F*( (2*x-1)**(i+1) - 2*i*x*(1-x)/(2*x-1)**(1-i) )  + K*\sum_{i=0}^15 Ai/F*(2*x-1)**i*(2*(i+1)*x**2 -2*(i+1)*x + 1) )
where Ai are RK coefficients, K is an empirical skew factor
"""
K = 3.932999*10**(-2)
U0 = 3.407141
As = [ -2.244923*10**3, -2.090675*10**3, -6.045274*10**3, -6.046354*10**3, -1.395210*10**4, 4.928595*10**4, 5.768895*10**4, -2.706196*10**5, -2.623973*10**5, 6.954912*10**5, 4.805390*10**5, -8.818037*10**5, -4.500675*10**5, 4.255778*10**5, 1.278146*10**5]


def OCV_Plett(x, K, U0, As):
    R = 8.314
    T = 298
    F = 96485.0
    U = U0+ R*T/F*np.log((1-x)/x)
    for i in range(1, len(As)+1):
        A_now = As[i-1]
        U = U + A_now/F*( (2*x-1)**(i+1) - 2*i*x*(1-x)/(2*x-1)**(1-i) )  + K*A_now/F * (2*x-1)**i *(2*(i+1)*x**2 -2*(i+1)*x + 1) 
    return U

# read hysterisis data
working_dir = os.getcwd()
df = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
data = df.to_numpy()
# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
xs = 1.0-data[:,0]/169.91 # i.e. Li concentration, divided by the theoretical capacity of LFP 
OCV_true = data[:,1]
OCV_pred = OCV_Plett(xs, K, U0, As)
SOC = 1.0-xs

# save fitted results
np.savez("RK_Plett.npz", x=SOC, y=OCV_pred)
# can be load as data=np.load("RK_Plett.npz"), SOC = data['x'], OCV_pred_Plett = data['y']

# plot figure 
plt.figure(figsize=(5,4))
plt.plot(SOC, OCV_pred, 'r-', label="Plett fitted OCV")
plt.plot(SOC, OCV_true, 'b-', label="True OCV")
plt.xlim([0,1])
plt.ylim([1.5, 5.0])
plt.legend()
plt.xlabel("SOC")
plt.ylabel("OCV")
fig_name = "Pred.png" 
plt.show()
# plt.savefig(fig_name, bbox_inches='tight')
# plt.close()

# calculate RMSE of fitted SOC (U_pred_after_ct) and the true value  (U_true_value)
loss = np.sqrt(np.mean((OCV_true - OCV_pred)**2))
print("RMSE = %.4f" %(loss))
filename = "RMSE_%.4f" %(loss)
with open(filename, 'w') as fin:
    fin.write("RMSE = %.4f" %(loss))
        
