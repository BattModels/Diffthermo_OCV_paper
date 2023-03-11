import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

"See if x is distributed evenly"


# read hysterisis data
df = pd.read_csv("Discharge_NMat_Fig2a_new.csv",header=None)
data = df.to_numpy()

# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
x = 1.0-data[:,0]/169.91 # i.e. Li concentration, divided by the theoretical capacity of LFP # TODO BUG 160 is guessed number, real LFP capacity is 169.91
mu = -data[:,1]*96485 # because -mu_e- = OCV*F, -OCV*F = mu
# convert to torch.tensor
x = x.astype("float32")
mu = mu.astype("float32")

y = [0]*len(x)
plt.plot(1-x, data[:,1], "*")
plt.show()

