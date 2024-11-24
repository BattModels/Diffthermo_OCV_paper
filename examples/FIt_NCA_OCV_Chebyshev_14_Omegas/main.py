from diffthermo.utils import train, write_ocv_functions


params_list = train(datafile_name='nca_ocp_Kim2011_data.csv', 
                        number_of_Omegas=14, 
                        polynomial_style = "Chebyshev",
                        learning_rate = 1000.0, 
                        total_training_epochs = 10000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-100*5000,-50*5000], 
                        Omegas_rand_range=[-100*100,100*100],
                        records_y_lims = [3.0,4.5])
write_ocv_functions(params_list, polynomial_style = "Chebyshev")



