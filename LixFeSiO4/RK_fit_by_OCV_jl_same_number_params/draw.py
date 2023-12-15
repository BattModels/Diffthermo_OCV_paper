import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# run julia
print("Running julia fit...")
os.system("rm PredictedOCV.csv RMSE_* RK_julia.npz Pred.png")
os.system("julia fit_rk.jl")
print("Julia running complete. ") 

# read true data
df = pd.read_csv("Discharge_NMat_Fig1a.csv",header=None)
data = df.to_numpy()
# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
SOC =  (data[:,0] - 1.0) # i.e. Li concentration
OCV_true = data[:,1]

# read data fitted by julia OCV.jl
df = pd.read_csv("PredictedOCV.csv",header=None)
data = df.to_numpy()
SOC_pred = data[:,0]
OCV_pred = data[:,1]
np.savez("RK_julia.npz", x=SOC_pred, y=OCV_pred) # can be load as data=np.load("RK_julia.npz"), SOC = data['x'], OCV_pred_RK = data['y']



# plot figure 
plt.figure(figsize=(5,4))
# plt.figure(figsize=(100,80))
# plot the one before common tangent construction
plt.plot(SOC_pred, OCV_pred, 'r-', label="OCV.jl Fitted OCV")
plt.plot(SOC, OCV_true, 'b-', label="True OCV")
plt.xlim([0,1])
# plt.ylim([1.5, 5.0]) 
plt.legend()
plt.xlabel("SOC")
plt.ylabel("OCV")
fig_name = "Pred.png" 
plt.savefig(fig_name, bbox_inches='tight')
plt.close()

# calculate RMSE of fitted SOC (U_pred_after_ct) and the true value  (U_true_value)
loss = np.sqrt(np.mean((OCV_true - OCV_pred)**2))
print("RMSE = %.4f" %(loss))
filename = "RMSE_%.4f" %(loss)
with open(filename, 'w') as fin:
    fin.write("RMSE = %.4f" %(loss))
        
