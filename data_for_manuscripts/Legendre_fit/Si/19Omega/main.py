from diffthermo.utils import train, write_ocv_functions


params_list = train(datafile_name='Si.csv', 
                        number_of_Omegas=19, 
                        polynomial_style = "Legendre",
                        learning_rate = 2000.0, 
                        total_training_epochs = 1000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-10*5000,-5*5000], 
                        Omegas_rand_range=[-10*100,10*100])
write_ocv_functions(params_list, polynomial_style = "Legendre")



