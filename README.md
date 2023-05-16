# README

This github repo contains all source code & run results for the paper "Thermodynamically-Consistent Open-Circuit Voltage Models Using Differentiable Thermodynamics".


LFP: containing the diffthermo model (referred to as "This work" in Figure 2), which learns the discharging OCV curve of LFP with only 4 RK params. The diffthermo model is locaed in the subfolder Discharge_4_RK_params/. Discharge_NMat_Fig2a_even_distribution.csv contains the digitized LFP data. Data is from https://www.nature.com/articles/nmat2730, Figure 2a. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. 


graphite: containing the diffthermo model which learns the discharging OCV curve of LFP with 7 RK params. In this folder, train.py is adapted so that it supports multiple miscibility gaps in OCV profile. Graphite OCV data from https://iopscience.iop.org/article/10.1149/2.015301jes, Figure 2. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. 



LixFeSiO4: containing the diffthermo model which learns the discharging OCV curve of LixFeSiO4 with 6 RK params. Data is from https://www.nature.com/articles/nmat2730, Figure 1a. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. 


LCO: containing the diffthermo model which learns the discharging OCV curve of LCO with x RK params. Data is from https://iopscience.iop.org/article/10.1149/1.1503075, Figure 4a. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. 


LMP: containing the diffthermo model which learns the discharging OCV curve of LiMnPO4 with x RK params. Data is from https://www.mdpi.com/2313-0105/4/3/39, Figure 1. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. 


LMFP: containing the diffthermo model which learns the discharging OCV curve of LiMn0.5Fe0.5PO4 with x RK params. Data is from https://www.mdpi.com/2313-0105/4/3/39, Figure 1. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. 


NCA: Data from https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/input/parameters/lithium_ion/data/nca_ocp_Kim2011_data.csv


NCO: Data from https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/input/parameters/lithium_ion/data/nco_ocp_Ecker2015.csv


NMC: data from https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/input/parameters/lithium_ion/data/nmc_LGM50_ocp_Chen2020.csv



