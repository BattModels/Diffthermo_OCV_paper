import numpy as np
import matplotlib.pyplot as plt
import pybamm
from pybamm import exp, log, tanh, constants, Parameter, ParameterValues
import pandas as pd

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

# import LFP dataset from AboutEnergy
parameter_values = pybamm.ParameterValues.create_from_bpx("lfp_18650_cell_BPX.json")

# customize parameter values, line 53-56 updated according to SOH_calc 
parameter_values["Positive electrode OCP [V]"] = LFP_OCP
del parameter_values["Current function [A]"]
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = 27318.724867850222
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = 797.8425247911586
# parameter_values['Nominal cell capacity [A.h]'] = 2.1988774587373907
# parameter_values["Typical current [A]"] = 2.1988774587373907



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
    npz_name = "Monotonic_RK_LFP_c_rate_%.4f.npz" %(c_rate)
    np.savez(npz_name, t=t_sol, V=V_sol, rmse = rmse)
    return sol

temperature = 25
# C/20
c_rate = 0.05
filename = "LFP_25degC_Co20.csv"
solve(temperature, filename, c_rate, title="LFP C/20");
# C/2
c_rate = 0.5
filename = "LFP_25degC_Co2.csv"
solve(temperature, filename, c_rate, title="LFP C/2");
# 1C
c_rate = 1.0
filename = "LFP_25degC_1C.csv"
solve(temperature, filename, c_rate, title="LFP 1C");
# 2C
c_rate = 2.0
filename = "LFP_25degC_2C.csv"
solve(temperature, filename, c_rate, title="LFP 2C");

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
