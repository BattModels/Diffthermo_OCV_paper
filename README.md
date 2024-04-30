# diffthermo
diffthermo is a python library for fitting thermodynamically consistent OCV models. It takes OCV and SoC as input, automatically does all the fitting and returns OCV models in **PyBaMM** and **Matlab** formats. 


## Installation 
Install the environment first
```bash
conda create -n diffthermo python=3.11  torch=2.0.0 numpy=1.24 matplotlib pandas
```
After succesfully installed the environment, 
```bash
conda activate diffthermo 
```
download the github repo
```bash
git clone https://github.com/BattModels/Diffthermo_OCV_paper.git
```
```bash
cd Diffthermo_OCV_paper
```
and then install the package
```bash
pip install .
```

## Usage
It's as simple as only 3 lines of commands!
```python
from diffthermo.utils import train, write_ocv_functions
# fit the OCV function
params_list = train(datafile_name='graphite.csv', 
                        number_of_Omegas=6, 
                        learning_rate = 1000.0, 
                        total_training_epochs = 8000,
                        loss_threshold = 0.01,
                        G0_rand_range=[-10*5000,-5*5000], 
                        Omegas_rand_range=[-10*100,10*100])
# write the fitted OCV function in PyBaMM OCV function format
write_ocv_functions(params_list)
```
After the fitting, you can find your fitted PyBaMM OCV function in the file `fitted_ocv_functions.py`, and the MATLAB OCV function in the file `fitted_ocv_functions.m`, or directly find them in the terminal where you executed the fitting code. Copy and paste them into your own projects and that's it! Incredibly easy, isn't it?

See [`example_graphite_OCV.ipynb`](examples/example_graphite_OCV.ipynb) under folder `examples` as a quick example, and what do all the parameters for `train` function means. 

*Some quick notes:*
*1. If your fitted results does not look good, TRY adjusting `G0_rand_range` and `Omegas_rand_range` in `train` function. These two parameters control the initial guess of G0 and Omegas.* *Usually for an anode material,* `G0_rand_range=[-10*5000,-5*5000], Omegas_rand_range=[-10*100,10*100]` work well, *and for cathode material,* `G0_rand_range=[-100*5000,-50*5000], Omegas_rand_range=[-100*100,100*100]` work. 
*2. For* `number_of_Omegas`, *usually for a phase-separating material that has n phase separating regions, set number_of_Omegas to be 2n to 4n should work fine for most cases (e.g. for LFP which has one (n=1) phase separation region, number_of_Omegas=4 gives a good fit)*
*3. If the fitted OCV function has a loss value or RMSE, try to initialize `train` function for mutiple times, as the Omegas and G0 are randomly initialized each time when you call `train` function, and some initialization will lead to bad fittings*
*4. If after trying to adjust `G0_rand_range` and `Omegas_rand_range` and initializing `train` for multiple times you still get a large RMSE, then try to increase `number_of_Omegas`. As shown in [Figure 4 of the paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.3c03129), sometimes it does need 20+ parameters to get a good fit.*

If you want to know why exactly this fitting works, please refer to the paper ["Open-Circuit Voltage Models Should Be Thermodynamically Consistent"](https://pubs.acs.org/doi/10.1021/acs.jpclett.3c03129), or the code walk-through [recording](https://drive.google.com/file/d/1PhCyvpmG28VjrClAviWHXVTnlqQScGIM/view?usp=sharing). 


## Folders In This Repo
diffthermo: the source code. 

examples: example on how to fit a thermodynamically consistent OCV functions with a graphite OCV dataset. The jupyter notebook file `example_graphite_OCV.ipynb` explains how to use the package using the graphite OCV as an example. 

data_for_manuscripts: contains all source code & run results for the paper "Open-Circuit Voltage Models Should Be Thermodynamically Consistent".

pybamm_OCV_functions: contains all the 12 fitted thermodynamically consistent OCV functions, implemented in PyBamm. You can get your OCV model in **Matlab** by running the fitting process and check the output file `fitted_ocv_functions.m`.


## Cite this work
If you find this repo useful in your research, please cite this work as follows (BibTex Format):
```
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
```




