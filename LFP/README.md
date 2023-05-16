# README

Discharge_4_RK_params: containing the diffthermo model (referred to as "This work" in Figure 2), which learns the discharging OCV curve of LFP with only 4 RK params. Discharge_NMat_Fig2a_even_distribution.csv contains the digitized LFP data. Data is from https://www.nature.com/articles/nmat2730, Figure 2a. Data digitized by WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. Run `python3 train.py` to see the results (install pytorch first). 

RK_fit_by_OCV_jl: containing the Monotonic RK model results (referred to as "Monotonic RK" in Figure 2). Run `python3 draw.py` to get the results (fit_rk.jl will be called in draw.py which runs the model fitting process).

RK_fit_python: containing the Regular RK model results (referred to as "Regular RK" in Figure 2). Run `python3 rk_fit_least_params_changing_Nmat_Fig2a.py` to see the results.

RK_fit_from_Plett_2015: containing the model from Plett text book (referred to as "Skewed RK" in Figure 2), run `python3 draw.py` to get the results.

Discharge_4_RK_params_splines: deprecated. Nothing important there.

pybamm_run: contains the pybamm simulation results shown in Figure 3. Refer to README.md in the folder to see details.

draw_all.py: collect results from all 4 OCV models and draw Figure 2 of the manuscript.
