import numpy as np
import matplotlib.pyplot as plt
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues



parameter_values = pybamm.ParameterValues("Prada2013")
# see these modifications at https://github.com/pybamm-team/PyBaMM/commit/eabb72040892b964907e01cdc09130a7e25a1489
parameter_values["Current function [A]"] = 1.1  # 1.1 originally
parameter_values["Typical current [A]"] = 1.1
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 13584.0
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 35.0
parameter_values["Positive electrode porosity"] = 0.26
parameter_values["Positive electrode active material volume fraction"] = 0.5846
parameter_values["Positive particle radius [m]"] = 5.00e-8


model = pybamm.lithium_ion.DFN()
experiment = pybamm.Experiment(
    ["Discharge at 1C for 1 hours or until 2.5 V"] # or until 2.5 V
)

sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
sim.solve(initial_soc=1.0)


# plot the results
solution = sim.solution
t = solution["Time [s]"]
A = solution['Current [A]']
V = solution["Terminal voltage [V]"]

import matplotlib as mpl  
from matplotlib.ticker import FormatStrFormatter
mpl.rc('font',family='Arial')
plt.figure(figsize=(5.5,4))
plt.plot(t.entries,V.entries,'b-',label="Prada2013")
plt.xlabel("Time [s]")
plt.ylabel("Terminal voltage [V]")
plt.xlim([0,t.entries.max()])
print(t.entries.max())
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.savefig('prada.png', dpi=200, bbox_inches='tight') 
plt.close()
