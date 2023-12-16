from diffthermo.utils import train, write_ocv_functions


params_list = train(datafile_name='graphite.csv', 
                    number_of_Omegas=6, 
                    learning_rate = 1000.0, 
                    total_training_epochs = 200,
                    loss_threshold = 0.01)
write_ocv_functions(params_list)



