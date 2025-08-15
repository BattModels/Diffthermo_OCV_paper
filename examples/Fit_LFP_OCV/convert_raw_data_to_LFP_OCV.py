import pandas as pd 
import os

# read data
df = pd.read_csv("Discharge_NMat_Fig2a_even_distribution.csv",header=None)
data = df.to_numpy()

# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
data[:,0] = 1.0-data[:,0]/169.91 # i.e. Li concentration, divided by the theoretical capacity of LFP 

df.to_csv("LFP.csv", index=False, header=False)

