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



# # check whether the curve is correct
# x = np.linspace(0.001, 0.999, 1000)
# OCP_func = parameter_values['Positive electrode OCP [V]']
# ocv = []
# for i in range(0, len(x)):
#     ocv.append(OCP_func(x[i]).value)
# plt.plot(x, ocv)
# plt.show()
# exit()


model = pybamm.lithium_ion.DFN()
experiment = pybamm.Experiment(
    ["Discharge at 1C for 1 hours"] # or until 2.0 V
)

sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
SoC_init = 1.0
sim.solve(initial_soc=SoC_init)


# plot the results
solution = sim.solution
t = solution["Time [s]"].entries
A = solution['Current [A]'].entries
V = solution["Terminal voltage [V]"].entries
SoC = SoC_init-solution['Discharge capacity [A.h]'].entries/parameter_values["Nominal cell capacity [A.h]"]


import matplotlib as mpl  
mpl.rc('font',family='Arial')
plt.figure(figsize=(5.5,4))
plt.plot(SoC, V,'b-',label="Prada2013")
plt.xlabel("SoC")
plt.ylabel("Terminal voltage [V]")
plt.xlim([0,1.0])
print(t.max())
print(V.max())
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.savefig('prada.png', dpi=200, bbox_inches='tight') 
plt.close()
