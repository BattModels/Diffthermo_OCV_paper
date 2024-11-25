from diffthermo.utils import train, write_ocv_functions
import numpy as np 
import pandas as pd 

df = pd.read_csv("LiMnPO4.csv",header=None)
data = df.to_numpy()
# Note that for LFP, the reaction during discharge is FP + Li = LFP, therefore the more Li you have in LFP, the lower the OCV will be, i.e. x = 1-SOC
df[0] = df[0]/170.87
df = df.sort_values(by=[0])
df.to_csv("LiMnPO4_soc.csv", index=False, header=False)

params_list = train(datafile_name='LiMnPO4_soc.csv', 
                        number_of_Omegas=5, 
                        polynomial_style = "Legendre",
                        learning_rate = 2000.0, 
                        total_training_epochs = 1000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-10*5000,-5*5000], 
                        Omegas_rand_range=[-10*100,10*100],
                        records_y_lims = [2.5,4.75])
write_ocv_functions(params_list, polynomial_style = "Legendre")



