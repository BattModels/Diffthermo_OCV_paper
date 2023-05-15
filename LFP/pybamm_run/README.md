# README

PyBamm simulation with the diffthermo OCP function.

The parameter set comes from https://github.com/About-Energy-OpenSource/About-Energy-BPX-Parameterisation. 

Before running this repo, you shall download data from https://github.com/About-Energy-OpenSource/About-Energy-BPX-Parameterisation, put LFP/data and LFP/lfp_18650_cell_BPX.json in this folder and rename data folder as data_from_AboutEnergy

First run `python3 SOH_calc.py` in order to get the corrected Initial concentration in negative/positive electrode & nominal cell capacity, copy the values into custom_OCV.py lines 55-56 and then run `python3 custom_OCV.py` with different C values (open the scripts at lines 139-154) to see the results. 


