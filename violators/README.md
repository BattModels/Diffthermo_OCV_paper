# README

Models that violates monotonicity condition (and thus the second law of thermo)



RK_Plett_LFP.npz: skewed RK for LFP, from Plett's textbook (Battery Management Systems, Volume I: Battery Modeling, Volume 1), violating region x=[0.08, 0.16]. Since Plett gives all the coefficients in the textbook, no digitizer was used. 

RK_Plett_graphite.npz: skewed RK for MCMB, from Plett's textbook, violating region x=[0.20, 0.60]

RK_Plett_LTO.npz: skewed RK for LTO, from Plett's textbook, violating region x=[0.00, 0.20]

RK_Plett_LCO.npz: skewed RK for LCO, from Plett's textbook, violating region x=[0.00, 0.20]

RK_Plett_LMO.npz: skewed RK for LMO, from Plett's textbook, violating region x=[0.04, 0.20]

RK_Plett_NMC.npz: skewed RK for NMC, from Plett's textbook, violating region x=[0.04, 0.40] and 

<!-- RK_Plett_NCA.npz: skewed RK for NCA, from Plett's textbook, violating region x=[0.10, 0.20]  # this fit is probably problematic itself, it's so off -->

# Data appeared in Figure 1:

Karthikeyan_Fig3_J_Power_Sources_2008.npz: 2 parameters Margules model for MCMB from Karthikeyan et al, Figure 3 (https://doi.org/10.1016/j.jpowsour.2008.07.077), violating region [0.40, 0.80]

RK_Plett_LFP.npz: skewed RK for LFP, from Plett's textbook (Battery Management Systems, Volume I: Battery Modeling, Volume 1), violating region x=[0.08, 0.16]. Since Plett gives all the coefficients in the textbook, no digitizer was used. 

Nejad_fig1a_J_Power_Sources_2016.npz: 8th-order polynomial for LFP from Nejad et al., figure 1(a) (http://dx.doi.org/10.1016/j.jpowsour.2016.03.042), violating region x=[0.775, 0.925]

Weng_Fig4_J_Power_Sources_2014.npz: Double exponential function for a LFP|graphite battery from Weng et al., figure 4 Model 2. (In fact Model 5, i.e. 6th-order polynomial, violates as well) (http://dx.doi.org/10.1016/j.jpowsour.2014.02.026), violating region [0.80, 0.90]


Pan_Fig8_Energy_2017.npz: 6-th order polynomial for a NMC|graphite battery from Pan et al, Figure 8 (http://dx.doi.org/10.1016/j.energy.2017.07.099), violating region [0.10, 0.24]

Yu_Fig2a_Energies_2021.npz: 4-th order polynomial for NMC from Yu et al, Figure 2(a) (https://doi.org/10.3390/en14071797), violating reegion [0.25, 0.55]



