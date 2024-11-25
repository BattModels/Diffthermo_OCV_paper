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
    monotonic RK fit for LFP
    sto: stochiometry 
    """
    Omegas = [-438911.8574,
     431938.0523,
     -438906.0302,
     431585.1246,
     -433173.8530,
     434485.6472,
     -583370.6929,
     516628.6304,
     1319111.5875,
     -1596177.9059,
     -11349662.8351,
     17202198.4685,
     34116403.1608,
    -66126903.0486,
    -44896154.6078,
    123587560.6177,
    -19620394.5200,
    -55304656.0503,
    77356452.8883,
    -95936617.2704,
    24813910.5488,
    10018772.9947,
    -77831588.0681,
    138654579.9193,
    -57170074.9350,
    52073048.1665,
    -6858950.8482,
    -43535695.9286,
    67414434.6420,
    -138515777.5941,
    74884556.4805,
    -76992835.9786,
    45605381.6377,
    11220413.6798,
    -33421019.6982,
    101903905.2078,
    -98118587.8394,
    129087494.7937,
    -76066114.3081,
    75715517.3753,
    -36413991.5972,
    -35819379.7246,
    37223044.7266,
    -119530909.5345,
    94885216.0688,
    -142101424.1563,
    101272337.0916,
    -45357783.3383,
    23541267.6999,
    179080648.2592,
    -138576624.8793,
    0.0000]
    U0 = 5.6460
    U = U0 + 8.314*298/96485.0*pybamm.log((1-sto)/sto) 
    for i in range(0, len(Omegas)):
        U = U + Omegas[i]/96485*((2*sto-1)**(i+1) - 2*i*sto*(1-sto)*(2*sto-1)**(i-1))
    return U



# # import LFP dataset Prada2013
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


# import LFP dataset from AboutEnergy
parameter_values = pybamm.ParameterValues.create_from_bpx("lfp_18650_cell_BPX.json")
# customize parameter values
parameter_values["Positive electrode OCP [V]"] = LFP_OCP

# # check AboutEnergy OCV model
# ocv_a = parameter_values['Positive electrode OCP [V]']
# xs = np.linspace(0.09, 0.95, 10000)
# ocv = []
# for x in xs:
#     ocv.append(ocv_a(x).value)
# plt.plot(xs, np.array(ocv))
# plt.show()

# Solve for "x_100", "y_100", "Q", "x_0", "y_0". Detailed description can be found at https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/models/electrode-state-of-health.ipynb
param = pybamm.LithiumIonParameters()
Vmin = 2.0
Vmax = 3.65
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

# # Extended code
# # First we solve for x_100 and y_100
# model = pybamm.BaseModel()
# x_100 = pybamm.Variable("x_100")
# y_100 = (Q_Li - x_100 * Q_n) / Q_p
# y_100_min = 1e-10
# x_100_upper_limit = (Q_Li - y_100_min*Q_p)/Q_n
# model.algebraic = {x_100: U_p(y_100, T_ref) - U_n(x_100, T_ref) - Vmax}
# model.initial_conditions = {x_100: x_100_upper_limit}
# model.variables = {
#     "x_100": x_100,
#     "y_100": y_100
# }
# sim = pybamm.Simulation(model, parameter_values=parameter_values)
# sol = sim.solve([0])
# x_100 = sol["x_100"].data[0]
# y_100 = sol["y_100"].data[0]
# for var in ["x_100", "y_100"]:
#     print(var, ":", sol[var].data[0])
# # Based on the calculated values for x_100 and y_100 we solve for x_0
# model = pybamm.BaseModel()
# x_0 = pybamm.Variable("x_0")
# Q = Q_n * (x_100 - x_0) #= Q_p * (y_0 - y_100)
# y_0 = y_100 + Q/Q_p
# model.algebraic = {x_0: U_p(y_0, T_ref) - U_n(x_0, T_ref) - Vmin}
# model.initial_conditions = {x_0: 0.01}
# model.variables = {
#     "Q": Q,
#     "x_0": x_0,
#     "y_0": y_0,
# }
# sim = pybamm.Simulation(model, parameter_values=parameter_values)
# sol = sim.solve([0])
# x_0 = sol["x_0"].data[0]
# y_0 = sol["y_0"].data[0]
# Q = sol["Q"].data[0] # unit A.h
# for var in ["Q", "x_0", "y_0"]:
#     print(var, ":", sol[var].data[0])
# print("\n\n\n")

# uodate params
print("Positive: ", y_100*parameter_values['Maximum concentration in positive electrode [mol.m-3]'])
print("Negative: ", x_100*parameter_values['Maximum concentration in negative electrode [mol.m-3]'])
print("Nominal cell capacity ", Q)


# results for AboutEnergy:
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 27318.724867850222
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 797.8425247911586
parameter_values['Nominal cell capacity [A.h]'] = 2.1988774587373907
parameter_values["Typical current [A]"] = 2.1988774587373907
