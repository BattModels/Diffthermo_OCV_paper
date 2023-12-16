import numpy as np
import matplotlib.pyplot as plt
import pybamm
import pandas as pd

# import LFP dataset from AboutEnergy
parameter_values = pybamm.ParameterValues.create_from_bpx("nmc_pouch_cell_BPX.json")

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
        var_pts=var_pts,          
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
    npz_name = "AboutEnergy_c_rate_%.4f.npz" %(c_rate)
    np.savez(npz_name, t=t_sol, V=V_sol, rmse = rmse)
    return sol

temperature = 25
# C/20
c_rate = 0.05
filename = "NMC_25degC_Co20.csv"
solve(temperature, filename, c_rate, title="LFP C/20");
# C/2
c_rate = 0.5
filename = "NMC_25degC_Co2.csv"
solve(temperature, filename, c_rate, title="LFP C/2");
# 1C
c_rate = 1.0
filename = "NMC_25degC_1C.csv"
solve(temperature, filename, c_rate, title="LFP 1C");
# 2C
c_rate = 2.0
filename = "NMC_25degC_2C.csv"
solve(temperature, filename, c_rate, title="LFP 2C");


