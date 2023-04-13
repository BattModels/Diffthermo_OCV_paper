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

# parameter_values = pybamm.ParameterValues("Prada2013")
# # see these modifications at https://github.com/pybamm-team/PyBaMM/commit/eabb72040892b964907e01cdc09130a7e25a1489
# parameter_values["Current function [A]"] = 1.1  # 1.1 originally
# parameter_values["Typical current [A]"] = 1.1
# parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 13584.0
# parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 35.0
# parameter_values["Positive electrode porosity"] = 0.26
# parameter_values["Positive electrode active material volume fraction"] = 0.5846
# parameter_values["Positive particle radius [m]"] = 5.00e-8
# # customize parameter values
# parameter_values["Positive electrode OCP [V]"] = LFP_OCP
# # solve init model
# model = pybamm.lithium_ion.DFN()
# c_rate = 0.5
# time = 1/c_rate
# experiment_text = "Discharge at %.4fC for %.4f hours" %(c_rate, time) #  or until 2.0 V
# experiment = pybamm.Experiment([experiment_text])
# sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
# sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
# model_sol = sim.solve()


# import LFP dataset Prada2013
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

# # import LFP dataset from AboutEnergy
# parameter_values = pybamm.ParameterValues.create_from_bpx("lfp_18650_cell_BPX.json")
# # customize parameter values
# parameter_values["Positive electrode OCP [V]"] = LFP_OCP

# # test
# parameter_values = pybamm.ParameterValues("Mohtat2020")

# Solve for "x_100", "y_100", "Q", "x_0", "y_0". Detailed description can be found at https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/models/electrode-state-of-health.ipynb
param = pybamm.LithiumIonParameters()
Vmin = 2.0
Vmax = 4.2
Q_n = parameter_values.evaluate(param.n.Q_init) # TODO what is this
Q_p = parameter_values.evaluate(param.p.Q_init) # TODO what is this
Q_Li = parameter_values.evaluate(param.Q_Li_particles_init) # TODO what is this
U_n = param.n.prim.U
U_p = param.p.prim.U
T_ref = param.T_ref
# solve for "x_100", "y_100", "Q", "x_0", "y_0"
esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
inputs={ "V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
esoh_sol = esoh_solver.solve(inputs)
for var in ["x_100", "y_100", "Q", "x_0", "y_0"]:
    print(var, ":", esoh_sol[var].data[0])

x_0 = esoh_sol["x_0"].data[0]
x_100 = esoh_sol["x_100"].data[0]
y_0 = esoh_sol["y_0"].data[0]
y_100 = esoh_sol["y_100"].data[0]
Q = esoh_sol["Q"].data[0] # unit A.h

# For LFP cathode (positive electrode), we need to update 
# 'Initial concentration in positive electrode [mol.m-3]', converted from y_100*Q_p
# Let
# thickness = d
# area = A
# total volume = V (= d*A)
# active material volume fraction = p
# rho is mass density of LFP
# total mass of LFP m = pV*rho
# molar mass of LFP M = 157.757 g/mole
# total mole of LFP, i.e. total mole of Li  n = m/M, should be equal to 3600*Q_p/F
# i.e. n = p*V*rho / M = 3600*Q_p/96485
# i.e. Initial concentration in positive electrode = y_100*n/V = y_100/V*3600*Q_p/96485
V = parameter_values['Electrode height [m]']*parameter_values['Electrode width [m]']*parameter_values['Positive electrode thickness [m]']
print("Positive: ",y_100/V*3600*Q_p/96485)
# print(y_0/V*3600*Q_p/96485)

# For anode, same thing applies
V = parameter_values['Electrode height [m]']*parameter_values['Electrode width [m]']*parameter_values['Negative electrode thickness [m]']
# print(x_0/V*3600*Q_n/96485)
print("Negative: ",x_100/V*3600*Q_n/96485)

# nominal cell capacity
print("Nominal cell capacity ", Q)

# results:
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 10234.014906833087
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 0.025586977415651448
parameter_values['Nominal cell capacity [A.h]'] = 1.092869235291763
