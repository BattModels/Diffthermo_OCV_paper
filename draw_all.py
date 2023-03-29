"""
Draw all results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# True OCV data from NMat Fig 2a, LFP
os.chdir("Discharge_4_RK_params")
df_discharge = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
df_discharge = df_discharge.sort_values(0)
discharge_data = df_discharge.to_numpy()
SOC_true = discharge_data[:,0]/169.91 # divided by the theoretical capacity of LFP
OCV_true = discharge_data[:,1]

# diffthermo fit
data=np.load("RK_diffthermo.npz")
SOC_diffthermo = data['x']
OCV_diffthermo = data['y']
data=np.load("RK_diffthermo_SOC_close_to_1.npz")
SOC_diffthermo_close_to_1 = data['x']
OCV_diffthermo_close_to_1 = data['y']
os.chdir("../")

# OCV.jl fit
os.chdir("RK_fit_by_OCV_jl")
data=np.load("RK_julia.npz")
SOC_julia = data['x']
OCV_julia = data['y']
os.chdir("../")

# regular RK fit
os.chdir("RK_fit_python")
data=np.load("RK.npz")
SOC_RK = data['x']
OCV_RK = data['y']
data=np.load("RK_with_close_to_1.npz")
SOC_RK_with_close_to_1 = data['x']
OCV_RK_with_close_to_1 = data['y']
os.chdir("../")

# Plett skewed RK
os.chdir("RK_fit_from_Plett_2015")
data=np.load("RK_Plett.npz")
SOC_Plett = data['x']
OCV_Plett = data['y']
os.chdir("../")

# diffthermo with splines smoothing
os.chdir("Discharge_4_RK_params_splines")
data=np.load("RK_diffthermo_splines.npz")
SOC_diffthermo_splines = data['x']
OCV_diffthermo_splines = data['y']
os.chdir("../")




# import matplotlib as mpl  
# from matplotlib.ticker import FormatStrFormatter
# mpl.rc('font',family='Arial')

# # figure
# plt.figure(figsize=(5.5,4))
# plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_diffthermo,OCV_diffthermo,'r.',label="This work", markersize=4)
# # plt.plot(SOC_diffthermo_splines,OCV_diffthermo_splines,'c-.',label="Diffthermo w/ splines", markersize=1.5)
# plt.plot(SOC_julia,OCV_julia,'b>',label="Monotonic RK", markersize=1.5)
# plt.plot(SOC_RK,OCV_RK,'g-.',label="Regular RK", markersize=1.5)
# plt.plot(SOC_Plett,OCV_Plett,'m.',label="Skewed RK", markersize=1.5)
# plt.legend(fontsize=10) 
# plt.xlim([0.0,1.0])
# # plt.ylim([-0.02,0.04])
# plt.xlabel("SOC", fontsize=14)
# plt.xticks(fontsize=14)
# plt.ylabel("OCV (V)", fontsize=14)
# plt.yticks(fontsize=14)
# # plt.tight_layout()
# plt.savefig('all_fit.png', dpi=200, bbox_inches='tight') 
# plt.close()

# # figure zoomed-in for Regular RK
# plt.figure(figsize=(2, 1.5))
# plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_RK,OCV_RK,'g-.',label="Regular RK", markersize=1.5)
# plt.xlim([0.78,0.92])
# plt.ylim([3.40,3.43])
# # plt.xlabel("SOC", fontsize=10)
# plt.xticks(fontsize=10)
# # plt.ylabel("OCV (V)", fontsize=10)
# plt.yticks(fontsize=10)
# # plt.tight_layout()
# plt.savefig('zoomed_in_RK.png', dpi=200, bbox_inches='tight') 
# plt.close()

# # figure zoomed-in for Plett
# plt.figure(figsize=(2, 1.5))
# ax = plt.gca()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_Plett,OCV_Plett,'m-',label="Skewed RK", markersize=1.5)
# plt.xlim([0.05,0.16])
# plt.ylim([3.37,3.43])
# # plt.xlabel("SOC", fontsize=10)
# plt.xticks(fontsize=10)
# # plt.ylabel("OCV (V)", fontsize=10)
# plt.yticks(fontsize=10)
# # plt.tight_layout()
# plt.savefig('zoomed_in_Plett.png', dpi=200, bbox_inches='tight') 
# plt.close()

# # figure for Regular RK & diffthermo outside fitted region
# plt.figure(figsize=(2, 1.5))
# ax = plt.gca()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_RK_with_close_to_1,OCV_RK_with_close_to_1,'g-.',label="Regular RK", markersize=1.5)
# plt.plot(SOC_diffthermo_close_to_1,OCV_diffthermo_close_to_1,'r.',label="This work", markersize=1.5)
# plt.xlim([0.92,0.96])
# plt.ylim([2.5,5.0])
# # plt.xlabel("SOC", fontsize=10)
# plt.xticks(fontsize=10)
# # plt.ylabel("OCV (V)", fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('Outside_fitted_region.png', dpi=200, bbox_inches='tight') 
# plt.close()





# calculate RMSE, R2-score and AIC of these fits
from sklearn.metrics import r2_score, mean_squared_error

def AIC(k, x_pred, x_real):
    """
    Akaike information criterion
    k: number of parameters
    x_pred: predicted result
    x_real: true result
    """
    n = len(x_pred)
    SSE = np.sum((x_pred-x_real)**2)
    sigma2_MLE = SSE/n
    aic = 2*k  + n*(np.log(2*np.pi) + np.log(sigma2_MLE)) + 1/sigma2_MLE*SSE
    return aic



# diffthermo
r2 = r2_score(OCV_true, OCV_diffthermo)
rmse = mean_squared_error(OCV_true, OCV_diffthermo, squared=False) # squared=False returns RMSE value
k = 5
aic_base = AIC(k, OCV_diffthermo, OCV_true) #0.0 #
aic_relative = AIC(k, OCV_diffthermo, OCV_true) - aic_base
print("Diffthermo: R2 = %.4f, RMSE = %.4f, relative AIC = %.4f" %(r2, rmse, aic_relative))
# # diffthermo with splines
# r2 = r2_score(OCV_true, OCV_diffthermo_splines)
# rmse = mean_squared_error(OCV_true, OCV_diffthermo_splines, squared=False) # squared=False returns RMSE value
# print("Diffthermo w/ splines: R2 = %.4f, RMSE = %.4f" %(r2, rmse))
# OCV.jl 
r2 = r2_score(OCV_true, OCV_julia)
rmse = mean_squared_error(OCV_true, OCV_julia, squared=False)
k = 53
aic_relative = AIC(k, OCV_julia, OCV_true) - aic_base
print("OCV_julia: R2 = %.4f, RMSE = %.4f, relative AIC = %.4f" %(r2, rmse, aic_relative))
# regular RK
r2 = r2_score(OCV_true, OCV_RK)
rmse = mean_squared_error(OCV_true, OCV_RK, squared=False)
k = 32
aic_relative = AIC(k, OCV_RK, OCV_true) - aic_base
print("Regular RK: R2 = %.4f, RMSE = %.4f, relative AIC = %.4f" %(r2, rmse, aic_relative))
# Plett skewed RK
r2 = r2_score(OCV_true, OCV_Plett)
rmse = mean_squared_error(OCV_true, OCV_Plett, squared=False)
k = 17
aic_relative = AIC(k, OCV_Plett, OCV_true) - aic_base
print("Plett: R2 = %.4f, RMSE = %.4f, relative AIC = %.4f" %(r2, rmse, aic_relative))


# # draw better figs
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# def format_axes(fig):
#     for i, ax in enumerate(fig.axes):
#         ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)

# # gridspec inside gridspec
# fig = plt.figure(figsize=(10,6.5))
# gs0 = gridspec.GridSpec(1, 1, figure=fig)
# gs00 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs0[0])
# ax1 = fig.add_subplot(gs00[:, 0:3])
# ax2 = fig.add_subplot(gs00[0, 3])
# ax3 = fig.add_subplot(gs00[1, 3])
# ax4 = fig.add_subplot(gs00[2, 3])
# # format_axes(fig)

# # all fit
# ax1.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# ax1.plot(SOC_diffthermo,OCV_diffthermo,'r.',label="This work", markersize=4)
# # ax1.plot(SOC_diffthermo_splines,OCV_diffthermo_splines,'c-.',label="Diffthermo w/ splines", markersize=1.5)
# ax1.plot(SOC_julia,OCV_julia,'b>',label="Monotonic RK", markersize=1.5)
# ax1.plot(SOC_RK,OCV_RK,'g-.',label="Regular RK", markersize=1.5)
# ax1.plot(SOC_Plett,OCV_Plett,'m.',label="Skewed RK", markersize=1.5)
# ax1.legend(fontsize=14) 
# ax1.set_xlim([0.0,1.0])
# # ax1.set_ylim([-0.02,0.04])
# ax1.set_xlabel("SOC", fontsize=14)
# # ax1.set_xticks(fontsize=14)
# ax1.set_ylabel("OCV (V)", fontsize=14)
# ax1.tick_params(axis='both', which='major', labelsize=14)
# # ax1.tick_params(axis='both', which='minor', labelsize=14)
# # ax1.set_yticks(fontsize=14)
# # ax1.tight_layout()


# # figure zoomed-in for Regular RK
# plt.figure(figsize=(2.5,2))
# plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_RK,OCV_RK,'g-.',label="Regular RK", markersize=1.5)
# plt.xlim([0.78,0.92])
# plt.ylim([3.40,3.43])
# plt.xlabel("SOC", fontsize=10)
# plt.xticks(fontsize=10)
# plt.ylabel("OCV (V)", fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.savefig('zoomed_in_RK.png', dpi=200) 
# plt.close()

# # figure zoomed-in for Plett
# plt.figure(figsize=(2.5,2))
# plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_Plett,OCV_Plett,'m-',label="Skewed RK", markersize=1.5)
# plt.xlim([0.08,0.16])
# plt.ylim([3.37,3.435])
# plt.xlabel("SOC", fontsize=10)
# plt.xticks(fontsize=10)
# plt.ylabel("OCV (V)", fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.savefig('zoomed_in_Plett.png', dpi=200) 
# plt.close()

# # figure for Regular RK outside fitted region
# plt.figure(figsize=(2.5,2))
# # plt.plot(SOC_true,OCV_true,'k-',label="True OCV", markersize=1.5)
# plt.plot(SOC_RK_with_close_to_1,OCV_RK_with_close_to_1,'g-.',label="Regular RK", markersize=1.5)
# plt.xlim([0.92,0.96])
# plt.ylim([2.7,4.1])
# plt.xlabel("SOC", fontsize=10)
# plt.xticks(fontsize=10)
# plt.ylabel("OCV (V)", fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('Regular_RK_outside_fitted_region.png', dpi=200, bbox_inches='tight') 
# plt.close()



# plt.show()