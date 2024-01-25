# diffthermo
diffthermo is a python library for fitting thermodynamically consistent OCV models.


## Installation 
Install the environment first
```bash
conda create -n diffthermo python=3.11  torch=2.0.0 numpy=1.24 matplotlib pandas
```
After succesfully installed the environment, 
```bash
conda activate diffthermo 
```
and then copy the diffthermo folder into your work directory.

## Usage
It's as simple as only 3 lines of commands!
```python
from diffthermo.utils import train, write_ocv_functions
# fit the OCV function
params_list = train(datafile_name='data.csv', 
                    number_of_Omegas = 6, 
                    learning_rate = 1000.0, 
                    total_training_epochs = 8000,
                    loss_threshold = 0.01)
# write the fitted OCV function in PyBaMM OCV function format
write_ocv_functions(params_list)
```
After the fitting, you can find your fitted PyBaMM OCV function in the file `fitted_ocv_functions.py`.
See `examples` for the files. If you want to learn more on the method, please refer to the paper "Open-Circuit Voltage Models Should Be Thermodynamically Consistent", https://pubs.acs.org/doi/10.1021/acs.jpclett.3c03129


## Folders In This Repo
diffthermo: the source code. 

examples: example on how to fit a thermodynamically consistent OCV functions with a graphite OCV dataset.

data_for_manuscripts: contains all source code & run results for the paper "Open-Circuit Voltage Models Should Be Thermodynamically Consistent".

pybamm_OCV_functions: contains all the 12 fitted thermodynamically consistent OCV functions, implemented in PyBamm. 


## Cite this work
If you find this repo useful in your research, please cite this work as follows (BibTex Format):

@article{doi:10.1021/acs.jpclett.3c03129,
author = {Yao, Archie Mingze and Viswanathan, Venkatasubramanian},
title = {Open-Circuit Voltage Models Should Be Thermodynamically Consistent},
journal = {The Journal of Physical Chemistry Letters},
volume = {0},
number = {0},
pages = {1143-1151},
year = {0},
doi = {10.1021/acs.jpclett.3c03129},
URL = {     
        https://doi.org/10.1021/acs.jpclett.3c03129
},
eprint = { 
        https://doi.org/10.1021/acs.jpclett.3c03129
}
}




