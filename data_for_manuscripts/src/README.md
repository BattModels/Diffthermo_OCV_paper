# README

diffthermo.py: contains all the important functions for diffthermo fit

train.py: a template for you to use. Make sure that you change line 24-27 in order to input filling fraction of Li and OCV, line 40-60 in order to add RK parameters and change them into nn.Parameters, line 65 and line 72 to declare all trainable parameters, and line 88 for total epochs. See the instructions in train.py for details. After modifications, run `python3 train.py` to fit. Note that the fitted value of Omegas and G0 are printed out in both log and bash records. For predictions, simply modify the total number of epochs to be 0. 

