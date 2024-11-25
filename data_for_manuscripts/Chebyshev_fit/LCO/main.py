from diffthermo.utils import train, write_ocv_functions


params_list = train(datafile_name='LCO_Carlier2012JES_Fig4a.csv', 
                        number_of_Omegas=9, 
                        polynomial_style = "Chebyshev",
                        learning_rate = 1000.0, 
                        total_training_epochs = 10000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-100*5000,-50*5000], 
                        Omegas_rand_range=[-100*100,100*100],
                        records_y_lims = [3.0,4.7])
write_ocv_functions(params_list, polynomial_style = "Chebyshev")



