"""
Adapted from 
https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/models/electrode-state-of-health.ipynb
"""

import pybamm
import matplotlib.pyplot as plt
import numpy as np
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

# custumized LFP OCV function 
def LFP_OCP(sto):
    """
    RK fit for LFP
    sto: stochiometry 
    """
    x_min_Nat_Mater_paper = 0.05948799345743927 # should be matched to 0.0
    x_max_Nat_Mater_paper = 0.9934071137200987 # should be matched to 1.0
    sto = (x_max_Nat_Mater_paper-x_min_Nat_Mater_paper)*sto + x_min_Nat_Mater_paper # streching the soc because Nat Mater paper has a nominal capacity of LFP of 160
    _eps = 1e-7
    # rk params
    G0 = -336668.3750 # G0 is the pure substance gibbs free energy 
    Omega0 = 128205.6641
    Omega1 = 61896.4375
    Omega2 = -290954.8125
    Omega3 = -164259.3750
    Omegas = [Omega0, Omega1, Omega2, Omega3]
    # phase coexistence chemical potential
    mu_coex = -328734.78125
    # phase boudary
    x_alpha = 0.08116829395294189453
    x_beta = 0.92593580484390258789
    is_outside_miscibility_gap = (sto<x_alpha) + (sto>x_beta)
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap * mu_outside + (1-is_outside_miscibility_gap) * mu_coex
    return -mu_e/96485.0

parameter_values = pybamm.ParameterValues("Prada2013")
# see these modifications at https://github.com/pybamm-team/PyBaMM/commit/eabb72040892b964907e01cdc09130a7e25a1489
parameter_values["Current function [A]"] = 1.1  # 1.1 originally
parameter_values["Typical current [A]"] = 1.1
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 13584.0
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 35.0
parameter_values["Positive electrode porosity"] = 0.26
parameter_values["Positive electrode active material volume fraction"] = 0.5846
parameter_values["Positive particle radius [m]"] = 5.00e-8
# customize parameter values
parameter_values["Positive electrode OCP [V]"] = LFP_OCP


model = pybamm.lithium_ion.DFN()
c_rate = 0.5
time = 1/c_rate
experiment_text = "Discharge at %.4fC for %.4f hours" %(c_rate, time) #  or until 2.0 V
experiment = pybamm.Experiment([experiment_text])
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
model_sol = sim.solve()



