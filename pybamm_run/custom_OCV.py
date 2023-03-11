import numpy as np
import matplotlib.pyplot as plt
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues


def LFP_OCP(sto):
    """
    RK fit for LFP
    sto: stochiometry 
    """
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
    # mu_e = G0 + is_outside_miscibility_gap * 8.314*298.0*log((sto+_eps)/(1-sto+_eps))
    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gap * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gap * mu_outside + (1-is_outside_miscibility_gap) * mu_coex
    return -mu_e/96485.0


# # check whether the curve is correct
# x = np.linspace(0.05, 0.999, 100000)
# ocv = []
# for i in range(0, len(x)):
#     ocv.append(LFP_OCP(x[i]).value)
# plt.plot(x, ocv)
# plt.show()
# exit()


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
del parameter_values["Current function [A]"]

model = pybamm.lithium_ion.DFN()

c_rate = 2.0
experiment_text = "Discharge at %.4fC for 10000 hours or until 2.5 V" %(c_rate)
experiment = pybamm.Experiment([experiment_text])
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
solver = pybamm.CasadiSolver() # root_method=pybamm.AlgebraicSolver(method='lm'), mode="fast with events",  extra_options_setup={"max_num_steps": 10000}
sim.solve(initial_soc=1.0, solver=solver) 


# # try solving up to the time the solver failed
# param = model.default_parameter_values
# timescale = param.evaluate(model.timescale)
# t_end = 0.01*timescale # 0.015
# t_eval = np.linspace(0, t_end, 100)
# sim = pybamm.Simulation(model, parameter_values=parameter_values)
# # solver = pybamm.ScikitsDaeSolver() 
# solver = pybamm.CasadiSolver(root_method='casadi', dt_max=600, return_solution_if_failed_early=True) # root_method=pybamm.AlgebraicSolver(method='lm')
# sim.solve(t_eval = t_eval, initial_soc=1.0, solver=solver) # , solver=pybamm.CasadiSolver(mode="safe", dt_max=0.01)


# plot the results
solution = sim.solution
t = solution["Time [s]"]
A = solution['Current [A]']
V = solution["Terminal voltage [V]"]

import matplotlib as mpl  
from matplotlib.ticker import FormatStrFormatter
mpl.rc('font',family='Arial')
plt.figure(figsize=(5.5,4))
plt.plot(t.entries,V.entries,'b-',label="Diffthermo")
plt.xlabel("Time [s]")
plt.ylabel("Terminal voltage [V]")
plt.xlim([0,t.entries.max()])
print(t.entries.max())
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.savefig('diffthermo.png', dpi=200, bbox_inches='tight') 
plt.close()

# save solution
npz_name = "custom_LFP_c_rate_%.4f.npz" %(c_rate)
np.savez(npz_name, t=t.entries, V=V.entries)
