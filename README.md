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
# fit the function
params_list = train(datafile_name='data.csv', 
                    number_of_Omegas = 6, 
                    learning_rate = 1000.0, 
                    total_training_epochs = 8000,
                    loss_threshold = 0.01)
# write the result in PyBaMM OCV function format
write_ocv_functions(params_list)
```
After the fitting, you can find your fitted PyBaMM OCV function in the file `fitted_ocv_functions.py`.
See `examples` for the files. If you want to learn more on the method, please refer to the paper "Open-Circuit Voltage Models Should Be Thermodynamically Consistent"


## Folders In This Repo
diffthermo: the source code. 

examples: example on how to fit a thermodynamically consistent OCV functions with a graphite OCV dataset.

data_for_manuscripts: contains all source code & run results for the paper "Open-Circuit Voltage Models Should Be Thermodynamically Consistent".

pybamm_OCV_functions: contains all the 12 fitted thermodynamically consistent OCV functions, implemented in PyBamm.






