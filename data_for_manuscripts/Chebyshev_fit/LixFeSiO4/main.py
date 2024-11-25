from diffthermo.utils import train, write_ocv_functions
import numpy as np 
import pandas as pd 

df = pd.read_csv("Discharge_NMat_Fig1a.csv",header=None)
data = df.to_numpy()
# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
df[0] = df[0]-1.0
df = df.sort_values(by=[0])
df.to_csv("LixFeSiO4.csv", index=False, header=False)

params_list = train(datafile_name='LixFeSiO4.csv', 
                        number_of_Omegas=6, 
                        polynomial_style = "Chebyshev",
                        learning_rate = 1000.0, 
                        total_training_epochs = 10000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-100*5000,-50*5000], 
                        Omegas_rand_range=[-100*100,100*100],
                        records_y_lims = [2.0,3.8])
write_ocv_functions(params_list, polynomial_style = "Chebyshev")



