"""
Adapted from 
https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/models/electrode-state-of-health.ipynb
"""

import pybamm
import matplotlib.pyplot as plt
import numpy as np
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues

# custumized graphite OCV function 

def graphite_OCP(sto):
    """
    diffthermo fit for graphite
    sto: stochiometry 
    """
    _eps = 1e-7
    # rk params
    Omega0 = -3809.5845 
    Omega1 = -4032.4138 
    Omega2 = 6000.8306 
    Omega3 = -11625.2646 
    Omega4 = -62671.4648 
    Omega5 = 25442.2031 
    Omega6 = 116366.1172 
    Omega7 = -26409.8652 
    Omega8 = -90384.7344 
    G0 = -11595.3965 

    Omegas = [Omega0, Omega1, Omega2, Omega3, Omega4, Omega5, Omega6, Omega7, Omega8]

    # phase boudary 0
    x_alpha_0 = 0.5630063414573669  
    x_beta_0 = 0.8214209079742432
    mu_coex_0 = -7810.95215 # phase coexistence chemical potential
    is_outside_miscibility_gap_0 = (sto<x_alpha_0) + (sto>x_beta_0)
    # phase boudary 1
    x_alpha_1 = 0.2831532061100006  
    x_beta_1 = 0.4294382333755493
    mu_coex_1 = -11124.01855 # phase coexistence chemical potential
    is_outside_miscibility_gap_1 = (sto<x_alpha_1) + (sto>x_beta_1)

    # whether is outside all gap
    is_outside_miscibility_gaps = is_outside_miscibility_gap_0 * is_outside_miscibility_gap_1

    mu_outside = G0 + 8.314*300.0*log((sto+_eps)/(1-sto+_eps))
    for i in range(0, len(Omegas)):
        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))
    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) * ((1-is_outside_miscibility_gap_0)*mu_coex_0 + (1-is_outside_miscibility_gap_1)*mu_coex_1)
    return -mu_e/96485.0



# import LFP dataset from AboutEnergy
parameter_values = pybamm.ParameterValues.create_from_bpx("nmc_pouch_cell_BPX.json")
# customize parameter values
parameter_values["Negative electrode OCP [V]"] = graphite_OCP

# # check AboutEnergy OCV model
# ocv_a = graphite_OCP # parameter_values['Negative electrode OCP [V]']
# xs = np.linspace(0.09, 0.95, 10000)
# ocv = []
# for x in xs:
#     ocv.append(ocv_a(x).value)
# plt.plot(xs, np.array(ocv))
# plt.show()
# exit()

# Solve for "x_100", "y_100", "Q", "x_0", "y_0". Detailed description can be found at https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/models/electrode-state-of-health.ipynb
param = pybamm.LithiumIonParameters()
Vmin = 2.7
Vmax = 4.2
Q_n = parameter_values.evaluate(param.n.Q_init) # TODO what is this? defined in https://pybamm.readthedocs.io/en/latest/_modules/pybamm/parameters/lithium_ion_parameters.html
Q_p = parameter_values.evaluate(param.p.Q_init) # TODO what is this
Q_Li = parameter_values.evaluate(param.Q_Li_particles_init) # TODO what is this
U_n = param.n.prim.U
U_p = param.p.prim.U
T_ref = param.T_ref

# solve for "x_100", "y_100", "Q", "x_0", "y_0"
esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
inputs={ "V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
print(inputs)
esoh_sol = esoh_solver.solve(inputs)
for var in ["x_100", "y_100", "Q", "x_0", "y_0"]:
    print(var, ":", esoh_sol[var].data[0])
x_0 = esoh_sol["x_0"].data[0]
x_100 = esoh_sol["x_100"].data[0]
y_0 = esoh_sol["y_0"].data[0]
y_100 = esoh_sol["y_100"].data[0]
Q = esoh_sol["Q"].data[0] # unit A.h


# uodate params
print("Positive: ", y_100*parameter_values['Maximum concentration in positive electrode [mol.m-3]'])
print("Negative: ", x_100*parameter_values['Maximum concentration in negative electrode [mol.m-3]'])
# print("Nominal cell capacity ", Q)

# results for AboutEnergy:
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 22339.49106054701
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 19774.140522338083
