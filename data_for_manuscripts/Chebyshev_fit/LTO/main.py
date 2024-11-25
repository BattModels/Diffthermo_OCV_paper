from diffthermo.utils import train, write_ocv_functions


params_list = train(datafile_name='LTO.csv', 
                        number_of_Omegas=4, 
                        polynomial_style = "Chebyshev",
                        learning_rate = 2000.0, 
                        total_training_epochs = 1000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-10*5000,-5*5000], 
                        Omegas_rand_range=[-10*100,10*100],
                        records_y_lims = [0.0,3.0])
write_ocv_functions(params_list, polynomial_style = "Chebyshev")



