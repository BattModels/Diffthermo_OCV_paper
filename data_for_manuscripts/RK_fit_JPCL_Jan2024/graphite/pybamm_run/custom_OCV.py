import numpy as np
import matplotlib.pyplot as plt
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues
import pandas as pd

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

# import NMC dataset from AboutEnergy
parameter_values = pybamm.ParameterValues.create_from_bpx("nmc_pouch_cell_BPX.json")

# customize parameter values, line 53-56 updated according to SOH_calc 
parameter_values["Negative electrode OCP [V]"] = graphite_OCP
del parameter_values["Current function [A]"]
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 22339.49106054701
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 19774.140522338083
# parameter_values['Nominal cell capacity [A.h]'] = 2.1235499318518993
# parameter_values["Typical current [A]"] = 2.1235499318518993


# model
model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
model.events = [] # Turn off model events check, as simulating a current trace measured form experiment may cause slight exceedance of voltage limits
solver = pybamm.CasadiSolver("fast", atol=1e-6, rtol=1e-6) # solver
solver._on_extrapolation = "warn" # Prevent solver failure if interpolant bounds are exceeded by a negligible amount
# mesh
submesh_types = model.default_submesh_types
submesh_types["negative particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)
submesh_types["positive particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)
var_pts = {"x_n": 16, "x_s": 16, "x_p": 16, "r_n": 16, "r_p": 16}

def solve(temperature, filename, c_rate, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # load data
    data = pd.read_csv(
        "data_from_AboutEnergy/validation/" + filename,
        comment="#",
    ).to_numpy()
    # split out time [s] vs voltage [V]
    voltage_data = data[:, [0, 2]]
    # split out time [s] vs current [A]
    current_data = data[:, [0, 1]]
    # create current interpolant
    timescale = parameter_values.evaluate(model.timescale)
    current_interpolant = pybamm.Interpolant(
        current_data[:, 0], -current_data[:, 1], timescale * pybamm.t, interpolator="linear")
    # set drive cycle and update temperature
    parameter_values.update({
        "Current function [A]": current_interpolant,
        "Ambient temperature [K]": 273.15 + temperature,
        "Initial temperature [K]": 273.15 + temperature,
    }, check_already_exists=False)
    # simulation 
    sim = pybamm.Simulation(
        model, 
        parameter_values=parameter_values,
        solver=solver,
        submesh_types=submesh_types,
        var_pts=var_pts
    )
    # solve
    sol = sim.solve()
    t_sol = sol["Time [s]"].entries
    V_sol = sol["Terminal voltage [V]"].entries
    # plot
    ax[0].plot(voltage_data[:, 0], voltage_data[:, 1], "--", label=f"Experiment ({temperature}\N{DEGREE SIGN})")
    ax[0].plot(t_sol, V_sol, "-", label=f"Model ({temperature}\N{DEGREE SIGN})")
    ax[1].plot(
        voltage_data[:, 0], 
        (sol["Terminal voltage [V]"](t=voltage_data[:, 0]) - voltage_data[:, 1]) * 1000,
    )
    rmse = np.sqrt(
        np.nanmean((voltage_data[:, 1] - sol["Terminal voltage [V]"](t=voltage_data[:, 0]))**2)
    ) * 1000
    print(f"RMSE = {rmse:.3f} mV \n")
    ax[1].text(0.8, 0.2, f"RMSE: {rmse:.3f} mV ({temperature}\N{DEGREE SIGN})",
               horizontalalignment='center',
               verticalalignment='center',
               transform = ax[1].transAxes,
              )  
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Voltage [V]")
    ax[0].legend()
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Error [mV]")
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    # save solution
    npz_name = "custom_NMC_c_rate_%.4f.npz" %(c_rate)
    np.savez(npz_name, t=t_sol, V=V_sol, rmse = rmse)
    return sol

temperature = 25
# C/20
c_rate = 0.05
filename = "NMC_25degC_Co20.csv"
solve(temperature, filename, c_rate, title="NMC C/20");
# C/2
c_rate = 0.5
filename = "NMC_25degC_Co2.csv"
solve(temperature, filename, c_rate, title="NMC C/2");
# 1C
c_rate = 1.0
filename = "NMC_25degC_1C.csv"
solve(temperature, filename, c_rate, title="NMC 1C");
# 2C
c_rate = 2.0
filename = "NMC_25degC_2C.csv"
solve(temperature, filename, c_rate, title="NMC 2C");

# model = pybamm.lithium_ion.DFN()
# c_rate = 0.5
# time = 1/c_rate
# # experiment_text = "Discharge at %.4fC for %.4f hours" %(c_rate, time) #  or until 2.0 V
# experiment_text = "Discharge at %.4fC until 2.0 V" %(c_rate) #  or until 2.0 V
# experiment = pybamm.Experiment([experiment_text])
# sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
# # solver = pybamm.CasadiSolver() 
# # SoC_init = 1.0
# # sim.solve(initial_soc=SoC_init, solver=solver) 
# sim.solve() 


# # ## smaller time steps for finer resolution of simulation
# # c_rate = 2.0
# # parameter_values["Current function [A]"] = 1.1*c_rate/169.97*160.0 # setting c rate, / 169.97 * 160.0 is because of the data from Nat Mater paper
# # param = model.default_parameter_values
# # timescale = param.evaluate(model.timescale)
# # t_end = 3600.0/c_rate*timescale # 0.015
# # t_eval = np.linspace(0.0, t_end, 100000000)
# # sim = pybamm.Simulation(model, parameter_values=parameter_values)
# # SoC_init = 1.0
# # sim.solve(t_eval = t_eval, initial_soc=SoC_init)

# # plot the results
# solution = sim.solution
# t = solution["Time [s]"].entries
# A = solution['Current [A]'].entries
# V = solution["Terminal voltage [V]"].entries
# # # TODO BUG this is WRONG!
# # SoC = SoC_init-solution['Discharge capacity [A.h]'].entries/parameter_values["Nominal cell capacity [A.h]"]
# # draw 
# import matplotlib as mpl  
# mpl.rc('font',family='Arial')
# plt.figure(figsize=(5.5,4))
# plt.plot(t, V,'b-',label="Diffthermo")
# plt.xlabel("Time [s]")
# plt.ylabel("Terminal voltage [V]")
# print(t.max())
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.legend()
# plt.show()
# # plt.savefig('diffthermo.png', dpi=200, bbox_inches='tight') 
# # plt.close()

# # # save solution
# # npz_name = "custom_LFP_c_rate_%.4f.npz" %(c_rate)
# # np.savez(npz_name, t=t, SoC=SoC, V=V)
